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
        lecture_lines = ["Use the library analogy.", "Query: What I search for.", "Key: Label on the book.", "Value: Information in the book.", "It is a matching game."]
        self.setup_layout("The Mechanism: Queries, Keys, and Values (Q, K, V)", lecture_lines)
        
        # Create assets
        q_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/label.svg", color="#00FFFF")
        k_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/label.svg", color="#00FFFF")
        v_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/label.svg", color="#00FFFF")
        
        q_label = Text("Q", color="#00FFFF").next_to(q_icon, ORIGIN)
        k_label = Text("K", color="#00FFFF").next_to(k_icon, ORIGIN)
        v_label = Text("V", color="#00FFFF").next_to(v_icon, ORIGIN)
        
        q_group = VGroup(q_icon, q_label)
        k_group = VGroup(k_icon, k_label)
        v_group = VGroup(v_icon, v_label)
        
        # Apply layout fixes (per constraints 40)
        self.place_at_grid(q_group, "B3", scale_factor=0.7)
        self.place_at_grid(k_group, "C3", scale_factor=0.7)
        self.place_at_grid(v_group, "D3", scale_factor=0.7)
        
        line_qk = Line(q_group.get_bottom(), k_group.get_top(), color=WHITE)
        line_kv = Line(k_group.get_bottom(), v_group.get_top(), color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"), FadeIn(q_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"), FadeIn(k_group), Create(line_qk))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"), FadeIn(v_group), Create(line_kv))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(Flash(q_icon), Flash(k_icon), Flash(v_icon))
        self.wait(2)
