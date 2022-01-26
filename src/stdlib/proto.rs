use super::*;

pub(crate) fn proto(vm: &mut VM, map: RantMapHandle) -> RantStdResult {
  vm.cur_frame_mut().write_value(map.borrow().proto().map_or(RantValue::Empty, RantValue::Map));
  Ok(())
}

pub(crate) fn set_proto(vm: &mut VM, (map, proto): (RantMapHandle, Option<RantMapHandle>)) -> RantStdResult {
  map.borrow_mut().set_proto(proto);
  Ok(())
}