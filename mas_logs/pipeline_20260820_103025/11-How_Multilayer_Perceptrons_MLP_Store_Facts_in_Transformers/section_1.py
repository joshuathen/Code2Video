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
        self.setup_layout("The Analogy: Library vs. Book Content", 
                          ["Attention mechanisms act like a library's index catalog.", 
                           "MLP layers store the actual knowledge contents.", 
                           "Like books, MLPs hold the facts within."])
        
        # Assets
        library_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        library_svg.set_color("#ADD8E6")
        book_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        book_svg.set_color("#FFFFFF")
        
        library_label = Text("Index", font_size=20, color="#ADD8E6")
        book_label = Text("Content", font_size=20, color="#FFFFFF")
        
        library_group = VGroup(library_svg, library_label).arrange(DOWN)
        book_group = VGroup(book_svg, book_label).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_in_area(library_group, 'B2', 'D2', scale_factor=0.6)
        self.place_in_area(book_group, 'B5', 'D5', scale_factor=0.6)
        self.play(FadeIn(library_group), FadeIn(book_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(WHITE)
        # Move book into library
        self.play(book_group.animate.move_to(library_group.get_center()),
                  book_group.animate.scale(0.5))
        self.wait(1)
