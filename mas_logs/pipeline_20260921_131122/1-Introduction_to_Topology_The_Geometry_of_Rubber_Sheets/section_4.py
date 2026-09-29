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
        self.setup_layout("The Euler Characteristic", [
            "Euler characteristic is constant.",
            "Formula: Vertices minus edges plus faces.",
            "Always equals two for convex polyhedra."
        ])
        
        # Define colors for V, E, F as per storyboard
        c_v = "#FF4500" # OrangeRed
        c_e = "#00CED1" # DarkTurquoise
        c_f = "#32CD32" # LimeGreen
        
        formula = MathTex(r"V - E + F = 2", font_size=42)
        formula.set_color_by_tex("V", c_v)
        formula.set_color_by_tex("E", c_e)
        formula.set_color_by_tex("F", c_f)
        self.place_in_area(formula, 'A2', 'C5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87CEEB"))
        self.play(FadeIn(formula))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#87CEEB"))
        labels = VGroup(
            Text("V: Vertices", font_size=20, color=c_v),
            Text("E: Edges", font_size=20, color=c_e),
            Text("F: Faces", font_size=20, color=c_f)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(labels, 'D3', scale_factor=0.7)
        self.play(Write(labels))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#87CEEB"))
        
        # Load asset
        cube_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        self.place_in_area(cube_svg, 'E3', 'F5', scale_factor=1.0)
        
        # Animate assets according to requirements
        self.play(cube_svg.animate.set_color(c_v)) # Animate vertices
        self.play(cube_svg.animate.set_color(c_e)) # Animate edges
        
        result = Text("= 2", font_size=36, color="#FFD700")
        result.next_to(formula, RIGHT)
        self.play(Write(result))
        self.play(Indicate(result))
        
        self.wait(2)
