for i in range(4):
  for j in range(4):
    print(i, j)
    if j == 2 and i == 1:
      break
  else:
    continue
  
  print("do")