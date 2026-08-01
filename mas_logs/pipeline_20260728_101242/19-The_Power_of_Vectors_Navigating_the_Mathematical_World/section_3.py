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
        self.setup_layout("Scalar Multiplication: The Growing Bee", [
            "Scalars can scale a vector's length or magnitude.",
            "Positive scalars stretch or shrink the vector's arrow.",
            "Negative scalars flip the vector to the opposite direction."
        ])
        
        # Colors
        v_color = "#00FFFF" # Cyan
        scalar_color = "#FFFF00" # Yellow
        negative_color = "#FF00FF" # Magenta

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(v_color)
        
        # Bee representation [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bee.svg]
        bee = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bee.svg")
        bee.set_color(scalar_color)
        self.place_at_grid(bee, "D4", scale_factor=0.6) # Issue 28: shift bee to D4
        
        bee_label = Text("Bee", font_size=18)
        self.place_at_grid(bee_label, "D5", scale_factor=1.0) # Issue 29: shift label to D5
        
        # Initial vector v
        v_start = self.grid["D4"]
        v_end = self.grid["C5"]
        v_arrow = Arrow(v_start, v_end, buff=0, color=v_color, stroke_width=4)
        v_label = MathTex("\\vec{v}", color=v_color)
        # Position label near C5 (the end of the vector)
        self.place_at_grid(v_label, "C6", scale_factor=0.8)
        
        self.play(FadeIn(bee), Write(bee_label))
        self.play(GrowArrow(v_arrow), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(scalar_color)
        
        # Scale to 2v
        # Start at D4, move twice the distance to B6 (D->C->B, 4->5->6)
        v_end_scaled = self.grid["B6"]
        v_label_scaled = MathTex("2\\vec{v}", color=scalar_color)
        # Position label above B6
        self.place_at_grid(v_label_scaled, "A6", scale_factor=0.8)
        
        self.play(
            v_arrow.animate.put_start_and_end_on(v_start, v_end_scaled).set_color(scalar_color),
            Transform(v_label, v_label_scaled),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(negative_color)
        
        # Flip to -v
        # Start at D4, move opposite to v (D->E, 4->3)
        v_end_flipped = self.grid["E3"]
        v_label_flipped = MathTex("-\\vec{v}", color=negative_color)
        # Issue 27: Position label at E3
        self.place_at_grid(v_label_flipped, "E3", scale_factor=0.8)
        
        self.play(
            v_arrow.animate.put_start_and_end_on(v_start, v_end_flipped).set_color(negative_color),
            Transform(v_label, v_label_flipped),
            run_time=2
        )
        self.wait(2)
