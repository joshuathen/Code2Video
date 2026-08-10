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
        self.setup_layout("Conclusion: Why it Matters", [
            "Superposition enables massive quantum parallelism.",
            "Qubits calculate all possibilities at once.",
            "A quantum computer sees the whole library instantly."
        ])
        
        # Elements
        # Using SVG asset for library
        library = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        sphere = Sphere(radius=1.5, color=BLUE, fill_opacity=0.3)
        dot = Dot(color=YELLOW)
        bloch = VGroup(sphere, dot)
        label_sp = Text("Superposition", font_size=24, color=YELLOW)
        label_meas = Text("Measurement", font_size=24, color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Using fix from issue 38
        self.place_in_area(bloch, 'C4', 'F6', scale_factor=0.5)
        self.place_at_grid(library, 'B4', scale_factor=0.5)
        self.play(Create(bloch), FadeIn(library), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        # Using fix from issue 39
        self.place_at_grid(label_sp, 'B4', scale_factor=0.7)
        self.play(Write(label_sp))
        self.play(Indicate(bloch))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        # Using fix from issue 40
        self.place_at_grid(label_meas, 'E4', scale_factor=0.7)
        self.play(FadeIn(label_meas))
        self.play(FadeOut(bloch), FadeOut(label_sp), FadeOut(label_meas), FadeOut(library))
        self.wait(1)
