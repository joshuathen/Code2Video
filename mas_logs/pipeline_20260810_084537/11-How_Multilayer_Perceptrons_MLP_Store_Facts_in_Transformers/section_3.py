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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["MLPs function as Key-Value memory systems.", "Expansion layer acts as the detecting Key.", "Projection layer inserts the factual Value."]
        self.setup_layout("Mechanism: Key-Value Memory Pairs", lecture_lines)
        
        # Define objects
        # Asset usage: /scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg and lock.svg
        key_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg", color="#FFD700")
        lock_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg", color="#FFFFFF")
        
        # Matrix placeholders
        key_matrix = Matrix([[1, 0], [0, 1]], color="#FFD700")
        val_matrix = Matrix([[0, 1], [1, 0]], color="#00FF00")
        query_dot = Dot(color="#FFFFFF")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg]
        self.place_in_area(key_matrix, 'A4', 'C6', scale_factor=0.5)
        self.play(Create(key_matrix))
        self.place_at_grid(key_icon, 'A4', scale_factor=0.3)
        self.play(FadeIn(key_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        # Layout fix applied
        self.place_in_area(val_matrix, 'D4', 'F6', scale_factor=0.5)
        self.play(Create(val_matrix))
        # Add query dot animation
        self.place_at_grid(query_dot, 'B3', scale_factor=1.0)
        self.play(FadeIn(query_dot))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        # Add lock icon
        self.place_at_grid(lock_icon, 'D4', scale_factor=0.3)
        self.play(FadeIn(lock_icon))
        self.wait(1)
