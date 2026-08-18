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
        self.setup_layout("Summary and Real-World Application", [
            "Interference captures phase and amplitude information.", 
            "Diffraction restores the original light field.", 
            "Holography enables high-density data storage."
        ])
        
        # Reveal lecture lines
        self.play(FadeIn(self.lecture))

        # === Animation for Lecture Line 1 ===
        # Recap: recording and reconstruction flow
        self.lecture[0].set_color("#FFFFFF")
        path = VGroup(
            Dot(color=WHITE),
            Line(start=self.grid['E1'], end=self.grid['F3'], color=WHITE),
            Dot(color=WHITE)
        )
        self.place_in_area(path, 'E1', 'F3', scale_factor=0.5)
        self.play(Create(path), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        # Data cube (Asset integration)
        self.lecture[1].set_color("#00FFFF")
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        cube.set_color("#00FFFF")
        self.place_at_grid(cube, 'F4', scale_factor=0.5)
        self.play(Create(cube), cube.animate.scale(1.2), run_time=1.5)
        self.play(cube.animate.scale(1/1.2), run_time=0.5) # Pulsate effect

        # === Animation for Lecture Line 3 ===
        # Microscope beam/technology (Asset integration)
        self.lecture[2].set_color("#FFFF00")
        microscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg")
        microscope.set_color("#FFFF00")
        self.place_at_grid(microscope, 'B4', scale_factor=0.6)
        beam = Line(start=microscope.get_bottom(), end=self.grid['D5'], color="#FFFF00")
        self.play(FadeIn(microscope), Create(beam), run_time=1.5)
        
        self.wait(2)
