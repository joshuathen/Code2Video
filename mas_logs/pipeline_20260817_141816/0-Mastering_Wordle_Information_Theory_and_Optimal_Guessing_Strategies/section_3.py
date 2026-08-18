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
        self.setup_layout("Algorithm in Action: Frequency Analysis", [
            "Frequency analysis identifies superior words.",
            "Common letters appear in optimal guesses.",
            "CRANE leverages high-frequency letter distribution."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show Bar Chart
        bar_chart = BarChart(
            values=[0.13, 0.08, 0.07, 0.07, 0.07],
            bar_names=["E", "A", "R", "I", "O"],
            y_range=[0, 0.15, 0.05],
            y_axis_config={"include_tip": False},
        )
        self.place_in_area(bar_chart, 'A3', 'C6', scale_factor=0.75)
        self.play(Create(bar_chart), self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Common Letters
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show CRANE heatmap
        letters = ["C", "R", "A", "N", "E"]
        heatmap = VGroup(*[Square(side_length=0.8, fill_opacity=0.6, fill_color=MAROON if l in ["R", "N", "A", "E"] else GREY) for l in letters])
        labels = VGroup(*[Text(l, font_size=36) for l in letters])
        
        for i, (sq, lbl) in enumerate(zip(heatmap, labels)):
            sq.move_to(self.grid[f'D{i+1}'])
            lbl.move_to(sq.get_center())
            
        crane_word = VGroup(heatmap, labels)
        self.place_at_grid(crane_word, 'D3', scale_factor=0.9)
        
        # Add summary box
        summary_box = Rectangle(width=4, height=1.5, color=WHITE).set_fill(BLUE, opacity=0.3)
        self.place_in_area(summary_box, 'E1', 'F6', scale_factor=0.85)

        self.play(FadeIn(crane_word), Create(summary_box), self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
