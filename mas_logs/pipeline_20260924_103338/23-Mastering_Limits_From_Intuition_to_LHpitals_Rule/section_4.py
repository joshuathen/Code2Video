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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Limits ask where we go, not where we are.",
            "Epsilon-delta provides the rigorous definition for limits.",
            "L'Hôpital computes values when basic algebra fails."
        ]
        self.setup_layout("Synthesis & Summary", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        synthesis_label = Text("Synthesis", color="#FFD700", font_size=24)
        self.place_at_grid(synthesis_label, 'B4', scale_factor=0.8)
        self.play(self.lecture[0].animate.set_color("#FFD700"), Write(synthesis_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        algo_label = Text("Algorithmic Loop", color="#00BFFF", font_size=24)
        
        # Visualizing iterative/recursive steps (B033)
        loop_graphic = VGroup(
            Square(side_length=0.5, color="#00BFFF"),
            Arrow(start=UP, end=DOWN, color="#00BFFF").scale(0.5)
        ).arrange(DOWN)
        
        # Combine into group per instructions
        loop_group = VGroup(algo_label, loop_graphic).arrange(DOWN)
        self.place_in_area(loop_group, 'C4', 'D5', scale_factor=0.7)
        
        self.play(self.lecture[1].animate.set_color("#00BFFF"), Write(algo_label), Create(loop_graphic))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        solved_label = Text("Solved", color="#32CD32", font_size=24)
        self.place_at_grid(solved_label, 'E5', scale_factor=0.8)
        
        # Visual anchor: Target (B007)
        target = Circle(radius=0.2, color="#32CD32").move_to(self.grid['E5'])
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/checkmark.svg]
        checkmark = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/checkmark.svg")
        checkmark.set_color("#32CD32").scale(0.5).move_to(target.get_center())
        
        self.play(
            self.lecture[2].animate.set_color("#32CD32"), 
            Write(solved_label), 
            Create(target), 
            FadeIn(checkmark)
        )
        self.wait(2)
