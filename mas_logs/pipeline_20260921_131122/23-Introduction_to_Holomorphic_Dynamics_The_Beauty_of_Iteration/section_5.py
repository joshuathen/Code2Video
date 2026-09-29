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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Real-world Application", [
            "Simple rules generate profound, infinite complexity.",
            "Dynamics has diverse practical computational uses.",
            "Fractals model realistic natural phenomena perfectly."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Recap the definition of complex dynamics.
        self.lecture[0].set_color("#FFFFFF")
        # Visual: Simple iterative process represented by a loop
        dot = Dot(color=BLUE)
        self.place_at_grid(dot, "B5", scale_factor=0.8)
        self.add(dot)
        self.play(Rotating(dot, radians=2*PI, run_time=2))

        # === Animation for Lecture Line 2 ===
        # Show a real-world fractal [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg]
        self.lecture[1].set_color("#00FF00")
        # Load SVG
        cloud = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg", color=GREEN)
        self.place_at_grid(cloud, "E5", scale_factor=0.7)
        self.play(FadeIn(cloud))

        # === Animation for Lecture Line 3 ===
        # Conclude with the power of simple iterative rules.
        self.lecture[2].set_color("#FFFF00")
        # Visual: Concluding signifier
        text_end = Text("Done!", font_size=36, color=YELLOW)
        self.place_at_grid(text_end, "F3", scale_factor=0.9)
        self.play(Write(text_end))
        self.wait(1)
