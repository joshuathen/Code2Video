from manim import *

# Fix: Set config to prevent the race condition in the LaTeX cleanup utility
config.no_latex_cleanup = True

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
        self.setup_layout("The Unified Notation Triangle", [
            "Power, root, and log are related.",
            "Visualize them as corners of a triangle.",
            "They all describe the same relationship."
        ])
        
        # Triangle vertices: b (base), x (exponent), y (result)
        # Using 3^4 = 81
        v_b = MathTex("b=3", color=WHITE)
        v_x = MathTex("x=4", color=WHITE)
        v_y = MathTex("y=81", color=WHITE)
        
        self.place_at_grid(v_b, "B3", scale_factor=1.2)
        self.place_at_grid(v_x, "E2", scale_factor=1.2)
        self.place_at_grid(v_y, "E4", scale_factor=1.2)
        
        triangle = Polygon(v_b.get_center(), v_x.get_center(), v_y.get_center(), color=WHITE)
        
        label_power = Text("Power", color="#00FFFF", font_size=20)
        label_log = Text("Log", color="#00FFFF", font_size=20)
        label_root = Text("Root", color="#00FFFF", font_size=20)
        
        self.place_at_grid(label_power, "C2", scale_factor=0.6)
        self.place_at_grid(label_log, "E3", scale_factor=0.6)
        self.place_at_grid(label_root, "C4", scale_factor=0.6)

        triangle_group = VGroup(triangle, v_b, v_x, v_y, label_power, label_log, label_root)
        self.place_in_area(triangle_group, "B2", "F5", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(Create(triangle), Write(v_b), Write(v_x), Write(v_y))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(Write(label_power), Write(label_log), Write(label_root))
        self.lecture[1].set_color(YELLOW)
        self.play(Rotate(triangle_group, angle=PI/6))

        # === Animation for Lecture Line 3 ===
        self.play(v_b.animate.set_color(RED), label_power.animate.set_color(RED))
        self.play(v_x.animate.set_color(GREEN), label_log.animate.set_color(GREEN))
        self.lecture[2].set_color(YELLOW)
