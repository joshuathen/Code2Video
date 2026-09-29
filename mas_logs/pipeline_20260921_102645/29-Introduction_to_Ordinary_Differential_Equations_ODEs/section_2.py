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
        self.setup_layout("Core Definition: What is an ODE?", [
            "An ODE relates a function to derivatives.",
            "Algebra snapshots are static equations.",
            "ODEs are flows of moving systems."
        ])
        
        # Assets
        faucet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/faucet.svg")
        river = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg")
        
        # === Animation for Lecture Line 1 ===
        ode_eq = MathTex(r"\frac{dy}{dx} = f(x, y)", color="#FFFFFF")
        ode_label = Text("ODE", font_size=24, color="#FFFFFF")
        self.place_in_area(ode_eq, 'B2', 'C4', scale_factor=0.9)
        self.place_at_grid(ode_label, 'A3', scale_factor=0.8)
        self.place_at_grid(faucet, 'A4', scale_factor=0.5)
        
        self.play(Write(ode_eq), FadeIn(ode_label), FadeIn(faucet))
        self.lecture[0].set_color("#00FFFF")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        static_eq = MathTex(r"x^2 + y^2 = 1", color="#FFFF00")
        static_label = Text("Static", font_size=24, color="#FFFF00")
        
        self.play(FadeOut(ode_eq), FadeOut(ode_label), FadeOut(faucet))
        self.place_in_area(static_eq, 'D2', 'E4', scale_factor=0.9)
        self.place_at_grid(static_label, 'C3', scale_factor=0.8)
        self.play(Write(static_eq), FadeIn(static_label))
        self.lecture[1].set_color("#FFFF00")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Create a simple vector field representation
        vector_field = VGroup(*[
            Arrow(start=self.grid[r+c], end=self.grid[r+c] + np.array([0.2, 0.2, 0]), buff=0, color="#00FF00", stroke_width=2)
            for r in "BCDE" for c in "2345"
        ])
        
        self.play(FadeOut(static_eq), FadeOut(static_label))
        self.place_at_grid(river, 'C3', scale_factor=0.5)
        self.play(Create(vector_field), FadeIn(river))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
