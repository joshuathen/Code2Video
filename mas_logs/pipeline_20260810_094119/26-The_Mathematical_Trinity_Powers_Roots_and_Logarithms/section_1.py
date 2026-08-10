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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introduction: The Three Perspectives", [
            "Powers, roots, and logs are three perspectives.",
            "They describe the same relationship: base, exponent, result.",
            "Think of this as a triangle of operations."
        ])

        # Create nodes using the asset
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg
        def create_node(text, color):
            node_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
            node_icon.set_color(color)
            label = Text(text, font_size=18, color=WHITE)
            return VGroup(node_icon, label).arrange(DOWN)

        power_node = create_node("Power", "#FF6600")
        root_node = create_node("Root", "#33CC33")
        log_node = create_node("Log", "#3399FF")

        # Position nodes (Triangle) as requested by VideoCritic
        self.place_at_grid(power_node, 'B5', scale_factor=0.7)
        self.place_at_grid(root_node, 'D4', scale_factor=0.7)
        self.place_at_grid(log_node, 'D6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(FadeIn(power_node), FadeIn(root_node), FadeIn(log_node))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # Create connecting lines
        line1 = Line(power_node.get_center(), root_node.get_center(), color=WHITE)
        line2 = Line(root_node.get_center(), log_node.get_center(), color=WHITE)
        line3 = Line(log_node.get_center(), power_node.get_center(), color=WHITE)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(Create(line1), Create(line2), Create(line3))
        self.wait(2)
