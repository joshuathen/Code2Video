from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Algorithm", [
            "Algorithm shifts disks in systematic order.",
            "Binary bits toggle as moves occur.",
            "This reveals a simple recursive structure."
        ])
        
        # Assets
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disks.svg")
        
        # Create recursion tree structure
        tree = VGroup()
        nodes = []
        for i in range(7):
            node = Circle(radius=0.15, color="#D3D3D3")
            nodes.append(node)
            tree.add(node)
            
        # Manually position nodes for tree-like structure
        nodes[0].move_to(self.grid["A4"])
        nodes[1].move_to(self.grid["C2"])
        nodes[2].move_to(self.grid["C6"])
        nodes[3].move_to(self.grid["E1"])
        nodes[4].move_to(self.grid["E3"])
        nodes[5].move_to(self.grid["E5"])
        nodes[6].move_to(self.grid["E6"])

        lines = VGroup(
            Line(nodes[0].get_center(), nodes[1].get_center(), color="#D3D3D3"),
            Line(nodes[0].get_center(), nodes[2].get_center(), color="#D3D3D3"),
            Line(nodes[1].get_center(), nodes[3].get_center(), color="#D3D3D3"),
            Line(nodes[1].get_center(), nodes[4].get_center(), color="#D3D3D3"),
            Line(nodes[2].get_center(), nodes[5].get_center(), color="#D3D3D3"),
            Line(nodes[2].get_center(), nodes[6].get_center(), color="#D3D3D3"),
        )
        
        algorithm_tree = VGroup(lines, tree)
        
        # Use assets as instructed
        disk_at_root = disk_icon.copy().scale(0.5).move_to(nodes[0].get_center())
        disk_at_right_branch = disk_icon.copy().scale(0.5).move_to(nodes[6].get_center())
        
        algorithm_tree.add(disk_at_root, disk_at_right_branch)
        
        self.place_at_grid(algorithm_tree, 'B4', scale_factor=0.9)
        self.add(algorithm_tree)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6347"))
        # Highlight left branch (#FF6347) for moving n-1 disks
        self.play(
            lines[0].animate.set_color("#FF6347"), 
            lines[2].animate.set_color("#FF6347"), 
            lines[3].animate.set_color("#FF6347")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        # Highlight right branch (#32CD32) for moving nth disk using asset
        self.play(
            lines[1].animate.set_color("#32CD32"), 
            lines[4].animate.set_color("#32CD32"), 
            lines[5].animate.set_color("#32CD32")
        )
        self.wait(1)
