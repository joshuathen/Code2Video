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
        lecture_lines = ["Data has a shape called a distribution.", "Every outcome is equally likely here.", "Many rolls form a uniform rectangle."]
        self.setup_layout("Prerequisite: The Concept of Distributions", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display text: 'Distributions describe data variability.' in #FFFFFF.
        text_label = Text("Distributions describe data variability.", font_size=24, color=WHITE)
        self.place_at_grid(text_label, 'B4', scale_factor=0.9)
        self.play(Write(text_label))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a histogram of a Normal distribution (#00FF00) fading in alongside the icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg].
        dice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")
        self.place_at_grid(dice_icon, 'D2', scale_factor=0.5)
        
        # Constructing a simple bar chart
        hist = VGroup(*[Rectangle(height=0.5 + i*0.2, width=0.4, color="#00FF00", fill_opacity=0.8) for i in range(5)]).arrange(RIGHT, buff=0.1)
        self.place_in_area(hist, 'C3', 'E5', scale_factor=1.2)
        
        self.play(FadeIn(hist), FadeIn(dice_icon))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the peak of the distribution in #FF0000.
        peak = hist[2]
        highlight = SurroundingRectangle(peak, color="#FF0000", buff=0.05)
        self.play(Create(highlight))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
