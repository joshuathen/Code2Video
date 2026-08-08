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
            "Transformer blocks feature attention and MLPs.",
            "Attention is the dynamic reader of context.",
            "MLPs act as static fact storage.",
            "Think of librarians reading specific books.",
            "Information resides permanently in the MLP."
        ]
        self.setup_layout("Prerequisite: The Two-Part Architecture", lecture_lines)
        
        # Use assets as requested
        librarian = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/librarian.svg")
        book = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        
        # Group with labels
        librarian_label = Text("Attention", font_size=20)
        lib_group = VGroup(librarian, librarian_label).arrange(DOWN)
        
        book_label = Text("MLP", font_size=20)
        book_group = VGroup(book, book_label).arrange(DOWN)
        
        # Positioning based on feedback: use C2-C3 for Attention, C5-C6 for MLP
        self.place_in_area(lib_group, 'C2', 'C3', scale_factor=0.8)
        self.place_in_area(book_group, 'C5', 'C6', scale_factor=0.8)
        
        arrow = Arrow(start=lib_group.get_right(), end=book_group.get_left(), color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(lib_group), FadeIn(book_group))
        self.lecture[0].set_color(WHITE)
        
        # === Animation for Lecture Line 2 ===
        self.play(lib_group.animate.set_color("#FFD700"))
        self.lecture[1].set_color("#FFD700")
        
        # === Animation for Lecture Line 3 ===
        self.play(book_group.animate.set_color("#FF4500"))
        self.lecture[2].set_color("#FF4500")
        
        # === Animation for Lecture Line 4 ===
        self.play(Create(arrow.set_color("#00FFFF")))
        self.lecture[3].set_color("#00FFFF")
        
        # === Animation for Lecture Line 5 ===
        self.play(Flash(lib_group, color=WHITE), Flash(book_group, color=WHITE))
        self.lecture[4].set_color(GOLD)
        
        self.wait(2)
