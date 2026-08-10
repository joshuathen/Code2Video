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
        lecture_lines = [
            "We use a library catalog analogy for retrieval.",
            "Query is what you want to find.",
            "Key is the label on the shelf.",
            "Value is the content inside the book.",
            "QKV enables intelligent information lookup."
        ]
        self.setup_layout("Prerequisite: Query, Key, and Value (QKV)", lecture_lines)
        
        # Load Assets
        library_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        book_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        
        # Define Q, K, V matrices
        q_mat = Matrix([["Q"]], color="#FF5733")
        k_mat = Matrix([["K"]], color="#33FF57")
        v_mat = Matrix([["V"]], color="#3357FF")
        
        qkv_group = VGroup(q_mat, k_mat, v_mat).arrange(RIGHT, buff=0.5)
        self.place_in_area(qkv_group, "B2", "D4", scale_factor=0.7) # Applying B048 scale and Issue 26/37 placement

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(qkv_group), FadeIn(library_icon.scale(0.5).next_to(qkv_group, UP)))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        self.play(q_mat.animate.set_color("#FFFF00"))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.play(k_mat.animate.set_color("#FFFF00"))
        self.play(q_mat.animate.set_color("#FF5733"))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        self.play(k_mat.animate.set_color("#33FF57"))
        self.lecture[3].set_color("#3357FF")

        # === Animation for Lecture Line 5 ===
        # Q * K^T simulation + Book Asset
        flash_rect = SurroundingRectangle(qkv_group, color="#FFFFFF", buff=0.2)
        self.play(Create(flash_rect))
        self.play(FadeOut(flash_rect))
        self.play(v_mat.animate.set_color("#33FF57"), FadeIn(book_icon.scale(0.5).next_to(v_mat, DOWN)))
        self.lecture[4].set_color("#FFFFFF")
        
        self.wait(2)
