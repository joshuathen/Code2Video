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
        lecture_lines = ["MLPs function as compressed, fuzzy databases.", "Knowledge is associative, not rigid.", "Connected weights allow robust inference."]
        self.setup_layout("Conclusion and Limitations", lecture_lines)
        
        # Mobjects
        mlp_repr = VGroup(*[Dot(radius=0.1) for _ in range(20)])
        mlp_repr.arrange_in_grid(4, 5, buff=0.2)
        # Apply fixes for issues 32/47
        self.place_in_area(mlp_repr, "A4", "C6", scale_factor=0.6)
        
        storage_label = Text("Storage Capacity", font_size=24, color="#FFC0CB")
        # Apply fixes for issues 33/48
        self.place_at_grid(storage_label, "D4", scale_factor=0.7)
        
        summary_text = Text("Robust & Associative", font_size=24, color="#FFFFFF")
        # Apply fixes for issues 34/49
        self.place_at_grid(summary_text, "F4", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        self.add(mlp_repr)
        self.play(FadeIn(mlp_repr))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFE0"))
        self.play(Write(storage_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#90EE90"))
        self.play(mlp_repr.animate.set_color("#A9A9A9"), FadeOut(storage_label), FadeIn(summary_text))
        self.wait(2)
