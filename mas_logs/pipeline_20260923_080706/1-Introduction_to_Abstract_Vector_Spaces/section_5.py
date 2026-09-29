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
        self.setup_layout("Summary and Synthesis", [
            "Vector spaces are defined by their structure.",
            "They serve as foundations for modern science.",
            "Abstraction empowers us to solve complex problems."
        ])
        
        # Create visual elements
        vs_structure = Text("Vector Space Structure", font_size=36, color=WHITE)
        modern_science = Text("Modern Science", font_size=30, color="#32CD32")
        complexity_solved = Text("Complexity Solved", font_size=30, color="#32CD32")
        
        # === Animation for Lecture Line 1 ===
        # Display the text 'Vector Space Structure'. Color: #FFFFFF.
        self.place_at_grid(vs_structure, 'C3')
        self.play(FadeIn(vs_structure))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Expand the text to fill the screen, showing it as a foundation. Color: #87CEEB.
        self.play(
            vs_structure.animate.scale(1.5).set_color("#87CEEB"),
            self.lecture[1].animate.set_color("#87CEEB")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade in 'Modern Science' and 'Complexity Solved' bullet points. Color: #32CD32.
        self.place_at_grid(modern_science, 'E2')
        self.place_at_grid(complexity_solved, 'E5')
        self.play(
            FadeIn(modern_science),
            FadeIn(complexity_solved),
            self.lecture[2].animate.set_color("#32CD32")
        )
        self.wait(2)
