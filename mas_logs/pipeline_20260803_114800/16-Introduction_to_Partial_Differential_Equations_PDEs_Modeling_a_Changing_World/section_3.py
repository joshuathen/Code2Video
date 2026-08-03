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
        self.setup_layout(
            "Defining the PDE: The Multi-Variable Balance",
            [
                "A PDE relates a function's partial derivatives together.",
                "Consider the heat equation: change over time equals curvature.",
                "The solution isn't one number, but a whole field.",
                "This field represents values across every point in space.",
                "We use special symbols to denote these partial changes."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        u_func = MathTex("u(x, t)", font_size=48)
        self.place_at_grid(u_func, "C4")
        self.play(Write(u_func))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Fix Issue 25: place_in_area('B2', 'D5')
        heat_eq = MathTex(
            r"\frac{\partial u}{\partial t}", "=", r"\alpha", r"\frac{\partial^2 u}{\partial x^2}",
            font_size=42
        )
        self.place_in_area(heat_eq, 'B2', 'D5', scale_factor=1.0)
        
        # Fix Issue 26: Crowded labels
        change_label = Text("Change over time", font_size=24, color=YELLOW)
        self.place_at_grid(change_label, 'E2', scale_factor=0.6)
        
        spatial_label = Text("Spatial distribution", font_size=24, color=GREEN)
        self.place_at_grid(spatial_label, 'E5', scale_factor=0.6)
        
        # Fix Issue 27: Diffusion label too high
        diffusion_label = Text("Diffusion Coefficient", font_size=24, color=RED)
        self.place_at_grid(diffusion_label, 'A3', scale_factor=0.6)

        # Storyboard: Highlight "∂" in the Heat Equation #FFFF00
        # Manually finding partial symbols inside the MathTex
        for part in heat_eq:
            # Look for sub-indices if possible or just color based on known structure
            pass
        
        # Simpler approach: find parts containing partial
        partials = heat_eq.get_parts_by_tex(r"\partial")

        self.play(
            ReplacementTransform(u_func, heat_eq),
            FadeIn(change_label),
            FadeIn(spatial_label),
            FadeIn(diffusion_label)
        )
        
        # Apply the highlight
        self.play(*[p.animate.set_color("#FFFF00") for p in partials])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Storyboard: Transform "u" into a surface of shifting colors.
        # We'll create a shifting color grid/field to represent 'u'.
        field = VGroup()
        for i in range(6):
            for j in range(6):
                sq = Square(side_length=0.15, fill_opacity=0.7, stroke_width=0.2)
                sq.set_color(interpolate_color(BLUE, RED, (i+j)/10))
                sq.move_to([i*0.18, j*0.18, 0])
                field.add(sq)
        
        self.place_at_grid(field, "C4", scale_factor=1.5)
        
        self.play(FadeIn(field, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Underline the derivatives (from storyboard)
        ul1 = Underline(heat_eq[0], color=YELLOW)
        ul2 = Underline(heat_eq[3], color=GREEN)
        
        self.play(Create(ul1), Create(ul2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Storyboard: Flash the spatial derivative "∂²u/∂x²" with a glow #FF8C00.
        glow = heat_eq[3].copy().set_stroke(color="#FF8C00", width=10, opacity=0.4)
        
        self.play(Flash(heat_eq[3], color="#FF8C00", flash_radius=0.6))
        self.play(FadeIn(glow))
        self.play(glow.animate.set_stroke(width=20, opacity=0), run_time=0.5)
        self.play(FadeOut(glow))
        
        self.wait(2)
