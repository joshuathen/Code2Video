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
            "Transformers use two distinct memory systems.",
            "Attention retrieves contextual information dynamically.",
            "MLPs store permanent factual knowledge.",
            "Think of MLPs as an internal encyclopedia.",
            "The brain analogy explains deep storage."
        ]
        self.setup_layout("The Analogy: The Brain vs. The Bookshelf", lecture_lines)
        
        # Load Assets
        bookshelf = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bookshelf.svg", color="#FFFFFF")
        book = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg", color="#00FF00")
        encyclopedia = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/encyclopedia.svg")
        
        # Create groups for bookshelf + books
        books = VGroup(*[book.copy() for _ in range(3)]).arrange(RIGHT, buff=0.1)
        bookshelf_group = VGroup(bookshelf, books).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.place_in_area(bookshelf_group, 'A2', 'B4', scale_factor=0.7)
        self.play(Create(bookshelf_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.place_at_grid(brain, 'D3', scale_factor=0.8)
        self.play(FadeIn(brain))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.play(bookshelf_group.animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        connecting_line = Line(brain.get_center(), bookshelf_group.get_center(), color="#00FF00")
        self.place_at_grid(connecting_line, 'C3', scale_factor=0.9)
        self.play(Create(connecting_line))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFA500"))
        self.play(FadeOut(bookshelf_group), FadeOut(brain), FadeOut(connecting_line))
        self.place_at_grid(encyclopedia, 'C3', scale_factor=1.2)
        self.play(FadeIn(encyclopedia))
