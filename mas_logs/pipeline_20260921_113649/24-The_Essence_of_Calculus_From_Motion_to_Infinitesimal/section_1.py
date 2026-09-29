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
        self.setup_layout("The Paradox of Instantaneous Motion", 
                          ["Motion changes constantly in every moment.", 
                           "How do we measure speed at one point?", 
                           "Average speed ignores the instant details."])
        
        # === Animation for Lecture Line 1 ===
        # Motion changes constantly in every moment.
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        # === Animation for Lecture Line 2 ===
        # How do we measure speed at one point?
        # Using asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg
        arrow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg", color="#FF0000")
        self.place_at_grid(arrow, 'B4', scale_factor=0.6)
        self.play(FadeIn(arrow), self.lecture[1].animate.set_color("#FF0000"))
        
        # === Animation for Lecture Line 3 ===
        # Average speed ignores the instant details.
        point = Dot(color="#00FF00", radius=0.1)
        self.place_at_grid(point, 'D5', scale_factor=0.7)
        # Offset point slightly by 0.5 units manually or by grid positioning adjustment
        # As grid is 1.0 apart, moving to E5 would be a clear offset, but constraints say D5.
        # Adding a label/offset as requested
        label = Text("t=0", font_size=18, color="#00FF00").next_to(point, DOWN, buff=0.2)
        
        self.play(FadeIn(point), FadeIn(label), self.lecture[2].animate.set_color("#00FF00"))
        
        self.wait(2)
