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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["We define three vectors for each word.", "Query: What am I looking for?", "Key: What relevant info do I offer?", "Value: The actual content I store.", "Think of it as a library search."]
        self.setup_layout("Core Concept: Query, Key, and Value", lecture_lines)
        
        # Define elements
        book_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        lib_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        
        q_box = Square(side_length=1.5, color="#FF9999")
        k_box = Square(side_length=1.5, color="#99FF99")
        v_box = Square(side_length=1.5, color="#9999FF")
        
        q_label = Text("Q", font_size=36, color="#FF9999")
        k_label = Text("K", font_size=36, color="#99FF99")
        v_label = Text("V", font_size=36, color="#9999FF")
        
        q_group = VGroup(q_box, q_label, book_icon.copy())
        k_group = VGroup(k_box, k_label, book_icon.copy())
        v_group = VGroup(v_box, v_label, book_icon.copy())

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(q_group), FadeIn(k_group), FadeIn(v_group))
        self.place_at_grid(q_group, 'C2', scale_factor=0.6)
        self.place_at_grid(k_group, 'C4', scale_factor=0.6)
        self.place_at_grid(v_group, 'C6', scale_factor=0.6)
        
        self.place_in_area(q_label, 'B2', 'B2', scale_factor=0.6)
        self.place_in_area(k_label, 'B4', 'B4', scale_factor=0.6)
        self.place_in_area(v_label, 'B6', 'B6', scale_factor=0.6)
        
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF9999")
        self.play(q_group.animate.scale(1.2))
        self.play(q_group.animate.scale(1/1.2))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#99FF99")
        self.play(k_group.animate.scale(1.2))
        self.play(k_group.animate.scale(1/1.2))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#9999FF")
        self.play(v_group.animate.scale(1.2))
        self.play(v_group.animate.scale(1/1.2))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FFFF")
        self.place_at_grid(lib_icon, 'E4', scale_factor=0.8)
        self.play(FadeIn(lib_icon))
        self.play(Indicate(q_group), Indicate(k_group), Indicate(v_group))
        self.wait(2)
