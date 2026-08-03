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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Linear Dependence: The Redundancy Trap", [
            "Linear dependence means one vector is redundant.",
            "It can be formed by combining other vectors.",
            "Redundant vectors do not expand the existing span."
        ])
        
        # Colors
        color_a = "#FFD700"  # Gold
        color_b = "#00BFFF"  # Deep Sky Blue
        color_c = "#FF4500"  # Orange Red
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_a))
        
        # Define vectors A and B
        # Origin at D2
        origin = self.grid["D2"]
        pos_a = self.grid["D4"]
        pos_b = self.grid["B2"]
        
        vec_a = Arrow(origin, pos_a, buff=0, color=color_a)
        label_a = MathTex(r"\vec{A}", color=color_a)
        self.place_at_grid(label_a, "E4", scale_factor=0.8)
        
        vec_b = Arrow(origin, pos_b, buff=0, color=color_b)
        label_b = MathTex(r"\vec{B}", color=color_b)
        # Fix for Issue 28: Adjust scale and keep at B1 as requested
        self.place_at_grid(label_b, 'B1', scale_factor=0.8)
        
        # Plane asset (replacing previous span_rect)
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg]
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        plane.set_stroke(WHITE, width=1).set_fill(WHITE, opacity=0.2)
        self.place_in_area(plane, "B2", "D4", scale_factor=1.0)
        
        self.play(Create(vec_a), Write(label_a))
        self.play(Create(vec_b), Write(label_b))
        self.play(FadeIn(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_c))
        
        pos_c = self.grid["B4"]
        vec_c = Arrow(origin, pos_c, buff=0, color=color_c)
        label_c = MathTex(r"\vec{C}", color=color_c)
        # Fix for Issue 29: Adjust placement of label_c closer to the tip
        self.place_at_grid(label_c, "A4", scale_factor=0.8)
        
        self.play(Create(vec_c), Write(label_c))
        self.wait(1)
        
        # Show C = A + B by shifting B
        vec_b_copy = vec_b.copy()
        self.play(
            vec_b_copy.animate.put_start_and_end_on(pos_a, pos_c).set_stroke(opacity=0.5),
            run_time=2
        )
        
        # Formula A + B = C
        formula = MathTex(
            r"\vec{A}", "+", r"\vec{B}", "=", r"\vec{C}",
            tex_to_color_map={r"\vec{A}": color_a, r"\vec{B}": color_b, r"\vec{C}": color_c}
        )
        # Fix for Issue 27: Use place_in_area for horizontal space
        self.place_in_area(formula, 'F2', 'F4', scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_b))
        
        # Highlight that C is inside the plane
        self.play(plane.animate.set_fill(color_b, opacity=0.4))
        self.play(Indicate(plane))
        self.play(Wiggle(vec_c))
        self.wait(2)
