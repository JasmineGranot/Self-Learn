from flask import Flask, request, jsonify
import json
from timeit import timeit


class Metadata:
    def __init__(self, num_of_vms):
        self._num_of_vm = num_of_vms
        self._num_of_queries = 0
        self._average = 0.0

    @property
    def num_of_vm(self):
        return self._num_of_vm

    @num_of_vm.setter
    def num_of_vm(self, val):
        self._num_of_vm = val

    @property
    def num_of_queries(self):
        return self._num_of_queries

    @num_of_queries.setter
    def num_of_queries(self, val):
        self._num_of_queries = val

    def add_query(self, time):
        new_avg = self._get_new_avg(self.num_of_queries, time)
        self._average = new_avg
        self.num_of_queries += 1

    def _get_new_avg(self, n, new_time):
        return ((n * self._average) + new_time) / (n + 1)

    def get_data_as_dict(self):
        return {'vm_count': self.num_of_vm, 'request_count': self.num_of_queries,
                'average_request_time': self._average}


class Consts:
    FILE_TO_LOAD_PATH = 'input-2.json'

    class VMConsts:
        VM_LIST = 'vms'
        VM_ID = 'vm_id'
        TAGS = 'tags'

        QUERY_PARAM_ID = 'vm_id'

    class FWConsts:
        FW_LIST = 'fw_rules'
        FW_ID = ' fw_id'
        SRC_TAG = 'source_tag'
        DST_TAG = 'dest_tag'


app = Flask(__name__)
app.config["DEBUG"] = True

with open(Consts.FILE_TO_LOAD_PATH) as f:
    data = json.load(f)

vm_in_data_list = data.get(Consts.VMConsts.VM_LIST, {})
fw_rules_in_data_list = data.get(Consts.FWConsts.FW_LIST, {})

meta = Metadata(len(vm_in_data_list))


@app.route('/', methods=['GET'])
def home():
    return "<h1>HOME PAGE</h1><p>This page should not be accessed .</p>"


@app.route('/api/v1/stats', methods=['GET'])
def api_stats():
    """
    Gets an API get request
    :return: a dictionary representing metadata object.
    """
    start_time = timeit()
    metadata = meta.get_data_as_dict()
    end_time = timeit()
    meta.add_query(end_time - start_time)
    return jsonify(metadata)


@app.route('/api/v1/attack', methods=['GET'])
def api_attack():
    """
    Gets an API get request with id of a virtual machine
    :return: virtual machines with an access to the given machine.
    """
    start_time = timeit()
    if Consts.VMConsts.QUERY_PARAM_ID in request.args:
        vm_id = request.args[Consts.VMConsts.QUERY_PARAM_ID]
    else:
        return "Error: No virtual machine id was provided. Please specify an vm_id."

    try:
        potential_threat_vm = _get_attackers(vm_id)

        return jsonify(_get_names_of_vms(potential_threat_vm))

    except ValueError as err:
        return f"Error: {err}"

    finally:
        end_time = timeit()
        meta.add_query(end_time - start_time)


def _get_names_of_vms(vm_list):
    return list(map(lambda d: d.get(Consts.VMConsts.VM_ID), vm_list))


def _get_attackers(vm_id):
    """
    :param vm_id: an id of a virtual machine to query. Assuming id is unique.
    :return: a list of virtual machines which has access to the queried virtual machine.
    """
    vm = _get_vm_by_id(vm_id)
    tags_with_access = _get_tags_set_by_firewall_rules(vm.get(Consts.VMConsts.TAGS))
    return _get_vm_by_tags(tags_with_access)


def _get_vm_by_tags(tags_set, vm_id=None, exclude_self=False):
    """
    :param tags_set: a set of tags
    :param vm_id: an id of specific virtual machine
    :param exclude_self: boolean: whether exclude the given vm
    :return: a list of virtual machines, each containing at least one of the given tags_list
    """
    vms_with_access_by_tag = filter(lambda d: set(d.get(Consts.VMConsts.TAGS)).intersection(tags_set), vm_in_data_list)

    if exclude_self and vm_id:
        vms_with_access_by_tag = filter(lambda d: d.get(Consts.VMConsts.VM_ID) != vm_id, vms_with_access_by_tag)

    return list(vms_with_access_by_tag)


def _get_vm_by_id(vm_id):
    """
    :param vm_id: an id of a virtual machine to query. Assuming id is unique.
    :return: a dictionary representing the vm given out of the data.
    """
    vm = list(filter(lambda d: d.get(Consts.VMConsts.VM_ID) == vm_id, vm_in_data_list))

    # make sure vm exist in the data
    if not vm:
        raise ValueError(f"Virtual machine with id {vm_id} was not found")

    return vm[0]


def _get_tags_set_by_firewall_rules(dst_tags):
    """
    :param dst_tags: a list of destination tags.
    :return: a set of source tags which has access to the destination tags provided according to firewall rules.
    """

    tags_with_access = set(
        map(lambda d: d.get(Consts.FWConsts.SRC_TAG),
            filter(lambda d: d.get(Consts.FWConsts.DST_TAG) in dst_tags, fw_rules_in_data_list)))

    return tags_with_access


app.run()
