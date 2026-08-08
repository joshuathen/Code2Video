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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Prerequisite Knowledge: Euler’s Formula",
            [
                "To solve this, we use Euler's formula for graphs.",
                "In a circle, regions equal edges minus vertices plus one.",
                "This formula calculates regions without manual counting."
            ]
        )

        # Colors for each animation stage
        color1 = "#90EE90"  # Light Green
        color2 = "#87CEFA"  # Light Sky Blue
        color3 = "#FFD700"  # Gold

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(color1))

        # Euler's Formula: V - E + F = 2
        # Apply fix for Issue 29: Start from B2 instead of B1 to avoid overlap (B021)
        euler_eq = MathTex("V", "-", "E", "+", "F", "=", "2", color=color1)
        self.place_in_area(euler_eq, "B2", "B6", scale_factor=1.5)

        # Labels for V, E, F at grid points below the formula
        v_label = MathTex("V: \\text{Vertices}", color=color1, font_size=24)
        e_label = MathTex("E: \\text{Edges}", color=color1, font_size=24)
        f_label = MathTex("F: \\text{Faces}", color=color1, font_size=24)

        self.place_at_grid(v_label, "C2")
        self.place_at_grid(e_label, "C4")
        self.place_at_grid(f_label, "C6")

        self.play(Write(euler_eq))
        self.play(FadeIn(VGroup(v_label, e_label, f_label), shift=UP * 0.3))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Transitions: Reset line 1, highlight line 2, fade out labels
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color2),
            FadeOut(v_label), FadeOut(e_label), FadeOut(f_label)
        )

        # Square example integration: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg]
        # (Issue 20) Load the asset and place it in the grid area.
        square_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        square_asset.set_color(color2)
        # Scale and position to match grid points D2-F5
        self.place_in_area(square_asset, "D2", "F5", scale_factor=2.0)
        
        # Corner points for the square based on grid coordinates
        p_tl = self.grid["D2"]
        p_tr = self.grid["D5"]
        p_bl = self.grid["F2"]
        p_br = self.grid["F5"]

        # Diagonal line (Issue 20 requirement: "with one diagonal line")
        diagonal = Line(p_tl, p_br, color=color2)

        # Vertices (dots)
        dots = VGroup(
            Dot(p_tl, color=color2, radius=0.08), 
            Dot(p_tr, color=color2, radius=0.08),
            Dot(p_bl, color=color2, radius=0.08), 
            Dot(p_br, color=color2, radius=0.08)
        )
        
        # Numerical labels for 4 vertices
        v_nums = VGroup(*[Text(str(i+1), font_size=20, color=color2) for i in range(4)])
        # Position using grid offsets or manual calculation relative to points
        v_nums[0].move_to(p_tl + UP*0.3 + LEFT*0.3)
        v_nums[1].move_to(p_tr + UP*0.3 + RIGHT*0.3)
        v_nums[2].move_to(p_bl + DOWN*0.3 + LEFT*0.3)
        v_nums[3].move_to(p_br + DOWN*0.3 + RIGHT*0.3)

        # Internal region labels
        f_labels = VGroup(
            Text("Face 1", font_size=18, color=color2),
            Text("Face 2", font_size=18, color=color2)
        )
        # Position face labels within internal regions
        self.place_at_grid(f_labels[0], "D4", scale_factor=1.0).shift(DOWN*0.5 + LEFT*0.3)
        self.place_at_grid(f_labels[1], "F3", scale_factor=1.0).shift(UP*0.5 + RIGHT*0.3)

        self.play(DrawBorderThenFill(square_asset))
        self.play(Create(diagonal), FadeIn(dots))
        self.play(Write(v_nums), Write(f_labels))
        
        # Calculation summary for the example
        # Apply fix for Issue 30: Move square_calc to E6 from F6 to avoid crowding
        square_calc = MathTex("V=4, E=5, R=2", color=color2, font_size=28)
        self.place_at_grid(square_calc, "E6")
        self.play(Write(square_calc))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Transitions: Reset line 2, highlight line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color3),
            FadeOut(square_calc),
            FadeOut(dots),
            FadeOut(square_asset),
            FadeOut(diagonal),
            FadeOut(v_nums),
            FadeOut(f_labels)
        )

        # Transformation: V - E + F = 2 -> R = E - V + 1
        # Apply fix for Issue 29: Start from B2 instead of B1
        transformed_eq = MathTex("R", "=", "E", "-", "V", "+", "1", color=color3)
        self.place_in_area(transformed_eq, "B2", "B6", scale_factor=1.5)
        
        # Explicit transformation of the formula
        self.play(Transform(euler_eq, transformed_eq))
        
        # Concluding label for the formula transformation
        # Apply fix for Issue 31: Move engine_label to E2-E5 to avoid overlaps with Row D
        engine_label = Text("Internal Regions Formula", font_size=24, color=color3)
        self.place_in_area(engine_label, "E2", "E5")
        
        self.play(Write(engine_label))
        self.wait(3)
