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
        self.setup_layout("Visualizing Attention Scores (Softmax)", 
                          ["Attention scores determine token relevance.", 
                           "High scores mean bright connections.", 
                           "Softmax maps similarity to importance."])
        
        # Consistent colors as per instruction
        score_color = "#AAAAAA"
        highlight_color = "#FF00FF"
        
        self.lecture[0].set_opacity(1)

        # === Animation for Lecture Line 1 ===
        # Represent scores as bars of different heights.
        bars = VGroup(*[
            Rectangle(width=0.5, height=h, fill_opacity=0.8, color=score_color, fill_color=score_color)
            for h in [1.5, 0.5, 2.0, 0.8]
        ]).arrange(RIGHT, buff=0.3)
        
        # Applying requested placement
        self.place_at_grid(bars, "C4", scale_factor=0.65)
        
        # Integrate placeholder asset as requested by task 17
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(icon, "B5", scale_factor=0.3)
        
        self.play(Create(bars), FadeIn(icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_opacity(0.5)
        self.lecture[1].set_opacity(1)
        # Highlight highest score bar with color #FF00FF.
        self.play(bars[2].animate.set_color(highlight_color))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_opacity(0.5)
        self.lecture[2].set_opacity(1)
        # Apply Softmax transformation to normalize bars to sum 1.
        new_heights = [0.2, 0.1, 0.5, 0.2]
        new_bars = VGroup(*[
            Rectangle(width=0.5, height=h*4, fill_opacity=0.8, color=score_color if i!=2 else highlight_color, fill_color=score_color if i!=2 else highlight_color)
            for i, h in enumerate(new_heights)
        ]).arrange(RIGHT, buff=0.3)
        
        # Applying requested placement
        self.place_in_area(new_bars, "B3", "D6", scale_factor=0.7)
        
        self.play(Transform(bars, new_bars))
        self.wait(2)
