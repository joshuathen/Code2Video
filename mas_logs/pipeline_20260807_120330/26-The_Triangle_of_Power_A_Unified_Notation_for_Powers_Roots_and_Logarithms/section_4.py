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
        # Data from storyboard and outline
        title = "Operation 1: Calculating the Power"
        lecture_lines = [
            "To find a power, look at the bottom right.",
            "The Base 'climbing' the Exponent produces the Result.",
            "Five squared yields twenty-five in our triangle."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        color_l1 = "#F39C12" # Orange
        color_l2 = "#3498DB" # Blue
        color_l3 = "#2ECC71" # Green
        
        # Assets
        triangle_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg"
        
        # === Animation for Lecture Line 1 ===
        # "To find a power, look at the bottom right."
        self.play(self.lecture[0].animate.set_color(color_l1))
        
        # Load and place the Triangle of Power SVG
        # Positioning it in area B2 to E6 to keep it centered and away from lecture text
        triangle_svg = SVGMobject(triangle_asset_path, color=WHITE)
        self.place_in_area(triangle_svg, "B2", "E6", scale_factor=2.0)
        
        # Define vertices for labeling based on typical equilateral triangle shape in that area
        # Top (Exponent): B4, Bottom-Left (Base): E2, Bottom-Right (Result): E6
        pos_base = self.grid["E2"]
        pos_exp = self.grid["B4"]
        pos_res = self.grid["E6"]
        
        label_base = MathTex(r"\text{Base}", color=WHITE)
        self.place_at_grid(label_base, "E2", scale_factor=0.7).shift(DOWN * 0.5)
        
        label_exp = MathTex(r"\text{Exponent}", color=WHITE)
        self.place_at_grid(label_exp, "B4", scale_factor=0.7).shift(UP * 0.5)
        
        label_res_text = MathTex(r"\text{Result}", color=WHITE)
        self.place_at_grid(label_res_text, "E6", scale_factor=0.7).shift(DOWN * 0.5)
        
        # Flashing "?" at the Result vertex
        res_q = MathTex("?", color=color_l1)
        self.place_at_grid(res_q, "E6", scale_factor=1.2)
        
        self.play(FadeIn(triangle_svg))
        self.play(Write(label_base), Write(label_exp), Write(label_res_text))
        self.play(FadeIn(res_q))
        self.play(Flash(res_q, color=color_l1, flash_radius=0.4))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The Base 'climbing' the Exponent produces the Result."
        self.play(self.lecture[1].animate.set_color(color_l2))
        
        # Bold/Highlight Base and Exponent labels
        self.play(
            label_base.animate.set_color(color_l2).scale(1.2),
            label_exp.animate.set_color(color_l2).scale(1.2)
        )
        
        # Visualize "climbing" path: Base -> Exponent -> Result
        # Using a simple line path for clarity
        path_points = [pos_base, pos_exp, pos_res]
        climb_path = VMobject(color=color_l2, stroke_width=4)
        climb_path.set_points_as_corners(path_points)
        
        climb_dot = Dot(color=color_l2)
        self.play(MoveAlongPath(climb_dot, climb_path), run_time=2, rate_func=linear)
        self.play(FadeOut(climb_dot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Five squared yields twenty-five in our triangle."
        self.play(self.lecture[2].animate.set_color(color_l3))
        
        # Values for the example 5^2 = 25
        val_5 = MathTex("5", color=WHITE)
        self.place_at_grid(val_5, "E2", scale_factor=0.9)
        
        val_2 = MathTex("2", color=WHITE)
        self.place_at_grid(val_2, "B4", scale_factor=0.9)
        
        # Final result value in color #2ECC71
        val_25 = MathTex("25", color=color_l3)
        self.place_at_grid(val_25, "E6", scale_factor=1.0)
        
        # Transition labels/symbols to concrete numbers
        self.play(
            Transform(label_base, val_5),
            Transform(label_exp, val_2),
            Transform(res_q, val_25),
            FadeOut(label_res_text),
            triangle_svg.animate.set_color(color_l3)
        )
        
        self.play(Indicate(val_25, color=color_l3, scale_factor=1.3))
        self.wait(2)

# Marking issues as under review
# update_issue(12, under_review=True, resolution_note="Generated Manim code for section_4 following storyboard and asset integration requirements. Implemented the Triangle of Power calculation visualization.")
# update_issue(24, under_review=True, resolution_note="Integrated triangle.svg asset as the central visualization for the section.")
