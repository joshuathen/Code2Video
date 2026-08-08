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
        self.setup_layout("The Query, Key, and Value Metaphor", [
            "Think of attention as a library search.",
            "Query is what you are searching for.",
            "Key is the tag; Value is the content."
        ])
        
        # Assets
        library_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        book_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        label_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/label.svg")
        
        # Q, K, V setup
        q_box = Rectangle(width=0.8, height=0.8, color="#FF5733", fill_opacity=0.5)
        k_box = Rectangle(width=0.8, height=0.8, color="#33FF57", fill_opacity=0.5)
        v_box = Rectangle(width=0.8, height=0.8, color="#3357FF", fill_opacity=0.5)
        
        q_label = Text("Q", font_size=20).move_to(q_box)
        k_label = Text("K", font_size=20).move_to(k_box)
        v_label = Text("V", font_size=20).move_to(v_box)
        
        q_group = VGroup(q_box, q_label)
        k_group = VGroup(k_box, k_label)
        v_group = VGroup(v_box, v_label)
        
        # === Animation for Lecture Line 1 ===
        # Display Query(Q), Key(K), and Value(V) boxes centered using [Asset: library.svg]
        self.place_at_grid(library_icon, 'B3', scale_factor=0.5)
        self.place_at_grid(q_group, 'B2', scale_factor=0.8)
        self.place_at_grid(k_group, 'B4', scale_factor=0.8)
        self.place_at_grid(v_group, 'B6', scale_factor=0.8)
        self.play(FadeIn(library_icon), Create(q_group), Create(k_group), Create(v_group))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Animate Query box sliding right toward set of Key boxes represented by [Asset: book.svg]
        self.place_at_grid(book_icon, 'D4', scale_factor=0.5)
        self.play(q_group.animate.move_to(self.grid['D3']), FadeIn(book_icon))
        self.lecture[1].set_color("#33FF57")
        
        # === Animation for Lecture Line 3 ===
        # Flash matching Key and Value boxes simultaneously in #FFFFFF with a [Asset: label.svg]
        self.place_at_grid(label_icon, 'E4', scale_factor=0.5)
        self.play(
            k_group.animate.set_color("#FFFFFF"),
            v_group.animate.set_color("#FFFFFF"),
            FadeIn(label_icon)
        )
        self.lecture[2].set_color("#3357FF")
        self.wait(1)
