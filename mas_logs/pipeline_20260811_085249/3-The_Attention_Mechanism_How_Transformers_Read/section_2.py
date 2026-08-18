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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Mechanics: Query, Key, and Value (Q, K, V)", [
            "Queries, Keys, and Values act as retrieval systems.",
            "Query matches Key to calculate attention scores.",
            "Attention scores scale the Value information."
        ])
        
        # Assets
        library_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(library_icon, 'A4', scale_factor=0.5)
        self.play(FadeIn(library_icon), Write(Text("Query, Key, Value", font_size=24, color=WHITE).next_to(library_icon, DOWN)))
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        q_label = Text("Q", color="#00FFFF")
        k_label = Text("K", color="#00FFFF")
        v_label = Text("V", color="#00FFFF")
        qkv_group = VGroup(q_label, k_label, v_label).arrange(RIGHT, buff=0.5)
        
        # === Animation for Lecture Line 2 ===
        self.place_in_area(qkv_group, 'B2', 'D5', scale_factor=0.9)
        self.play(FadeIn(qkv_group))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Animate Q, K interaction
        dot_product_visual = Line(q_label.get_right(), k_label.get_left(), color=YELLOW)
        self.place_at_grid(magnifying_glass, 'C3', scale_factor=0.4)
        self.play(Create(dot_product_visual), FadeIn(magnifying_glass))
        
        # === Animation for Lecture Line 3 ===
        self.play(Indicate(v_label))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        self.wait(2)
