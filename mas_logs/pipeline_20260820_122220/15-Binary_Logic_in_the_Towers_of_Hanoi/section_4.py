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
        self.setup_layout("Visualizing the Sequence", [
            "Binary counter matches disk movement.",
            "Observe the rhythmic bit-flip pattern.",
            "Disks follow binary logic perfectly."
        ])
        
        # Reveal lecture lines
        self.play(FadeIn(self.lecture))

        # Using SVGMobject for disk assets
        # Path: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg
        disk_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # Load disks once
        disk1 = SVGMobject(disk_path, color="#7B68EE").scale(0.3)
        disk2 = SVGMobject(disk_path, color="#7B68EE").scale(0.3)
        disk3 = SVGMobject(disk_path, color="#7B68EE").scale(0.3)
        
        nodes = VGroup(disk1, disk2, disk3).arrange(RIGHT, buff=0.8)
        
        # Edges
        edge1 = Line(disk1.get_right(), disk2.get_left(), color="#00FFFF")
        edge2 = Line(disk2.get_right(), disk3.get_left(), color="#00FFFF")
        tree_edges = VGroup(edge1, edge2)
        
        # Initial placement (using grid as per critic's advice for visibility)
        self.place_in_area(VGroup(nodes, tree_edges), "B4", "E6", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#7B68EE"), Create(nodes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Flash node 2
        flash = nodes[1].copy().set_color("#FF00FF").scale(1.2)
        self.play(self.lecture[1].animate.set_color("#FF00FF"), 
                  Flash(nodes[1], color="#FF00FF"),
                  nodes[1].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"), Create(tree_edges))
        self.wait(2)
