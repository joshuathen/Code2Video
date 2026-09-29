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
        self.setup_layout("The Concept of Composition", [
            "Consider two operations in a sequence.",
            "We apply matrix B, then matrix A.",
            "This composition is written as A(B(v))."
        ])
        
        # --- Create objects ---
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg]
        box_f = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg", color=WHITE)
        label_f = Text("B", font_size=24).move_to(box_f.get_center())
        f_group = VGroup(box_f, label_f)
        
        box_g = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg", color=WHITE)
        label_g = Text("A", font_size=24).move_to(box_g.get_center())
        g_group = VGroup(box_g, label_g)
        
        input_v = Dot(color=WHITE)
        label_v = Text("v", font_size=20).next_to(input_v, UP)
        v_group = VGroup(input_v, label_v)
        
        # Applying requested position fixes
        self.place_at_grid(f_group, 'B3', scale_factor=0.7)
        self.place_at_grid(g_group, 'B6', scale_factor=0.7)
        self.place_at_grid(v_group, 'C2', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(f_group), FadeIn(g_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFCC00")
        arrow1 = Arrow(start=v_group.get_right(), end=f_group.get_left(), color="#FFCC00")
        self.play(Create(arrow1), FadeIn(v_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFCC")
        arrow2 = Arrow(start=f_group.get_right(), end=g_group.get_left(), color="#00FFCC")
        self.play(Create(arrow2))
        self.wait(2)
