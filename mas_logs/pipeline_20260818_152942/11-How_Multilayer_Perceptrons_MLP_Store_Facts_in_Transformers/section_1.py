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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Attention layers act as a flexible search engine.",
            "MLP layers provide the model's long-term memory.",
            "Attention gathers relevant context for a token.",
            "MLP provides facts stored within its parameters.",
            "Together, they form a reasoning engine."
        ]
        self.setup_layout("The Analogy: The Brain vs. The Library", lecture_lines)
        
        # Load Assets
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        library = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        
        brain_label = Text("Attention (Brain)", font_size=20, color=WHITE)
        mlp_label = Text("MLP (Library)", font_size=20, color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_at_grid(brain_label, "B2", scale_factor=0.7)
        self.play(FadeIn(brain_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.place_at_grid(mlp_label, "B5", scale_factor=0.7)
        self.play(FadeIn(mlp_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.place_at_grid(brain, "E2", scale_factor=0.9)
        self.play(FadeIn(brain))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.place_at_grid(library, "E5", scale_factor=0.9)
        self.play(FadeIn(library))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.wait(1)
