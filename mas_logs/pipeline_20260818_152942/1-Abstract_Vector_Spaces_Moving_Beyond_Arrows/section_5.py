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
        self.setup_layout("Conclusion & Takeaway", [
            "Abstract vector spaces offer a universal language.",
            "They describe structure in complex systems.",
            "Linear algebra provides this powerful framework."
        ])
        
        # Elements
        summary = Text("Universal Language", font_size=32)
        list_items = VGroup(
            Text("Complex Systems", font_size=32),
            Text("Linear Structures", font_size=32),
            Text("Powerful Logic", font_size=32)
        )
        takeaway_content = Text("Abstract Vector Spaces", font_size=40, color=BLUE)
        
        # Placement
        self.place_in_area(summary, "B2", "C5", scale_factor=0.6)
        
        self.place_at_grid(list_items[0], "A4", scale_factor=0.7)
        self.place_at_grid(list_items[1], "B4", scale_factor=0.7)
        self.place_at_grid(list_items[2], "C4", scale_factor=0.7)
        
        self.place_in_area(takeaway_content, "E2", "F5", scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(summary), FadeIn(list_items[0]))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(list_items[1]), FadeIn(list_items[2]))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(takeaway_content))
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(FadeOut(self.lecture), FadeOut(self.title), FadeOut(summary), FadeOut(list_items), FadeOut(takeaway_content))
        self.wait(2)
