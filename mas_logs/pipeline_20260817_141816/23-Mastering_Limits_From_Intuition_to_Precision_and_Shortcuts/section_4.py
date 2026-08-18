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
        self.setup_layout("Synthesis & Summary", [
            "Limits help us understand behavior near points.",
            "Epsilon-delta provides the rigour we need.",
            "L'Hôpital's Rule offers an efficient shortcut."
        ])

        # Nodes
        node_1 = Circle(radius=0.4, color="#87CEEB", fill_opacity=0.6)
        label_1 = Text("Intuitive", font_size=16).next_to(node_1, DOWN, buff=0.1)
        self.place_at_grid(VGroup(node_1, label_1), "B2")

        node_2 = Circle(radius=0.4, color="#87CEEB", fill_opacity=0.6)
        label_2 = Text("Rigorous", font_size=16).next_to(node_2, DOWN, buff=0.1)
        self.place_at_grid(VGroup(node_2, label_2), "D4")

        node_3 = Circle(radius=0.4, color="#87CEEB", fill_opacity=0.6)
        label_3 = Text("Shortcut", font_size=16).next_to(node_3, DOWN, buff=0.1)
        self.place_at_grid(VGroup(node_3, label_3), "F4")
        
        rabbit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg")
        self.place_at_grid(rabbit, "B2", scale_factor=0.3)
        rabbit.move_to(node_1.get_center())

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(node_1), Write(label_1), FadeIn(rabbit))
        self.lecture[0].set_color("#87CEEB")

        # === Animation for Lecture Line 2 ===
        self.play(rabbit.animate.set_color("#FF8C00"), run_time=1)
        self.play(FadeIn(node_2), Write(label_2))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(node_3), Write(label_3))
        self.lecture[2].set_color("#FFFFFF")
        self.play(Flash(node_2, color="#FFFFFF"), Flash(node_3, color="#FFFFFF"))
