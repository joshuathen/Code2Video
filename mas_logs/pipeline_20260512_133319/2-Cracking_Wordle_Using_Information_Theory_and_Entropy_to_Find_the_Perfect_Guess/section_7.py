from manim import *
import numpy as np

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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initializing scene layout
        lecture_lines = [
            'Winning Wordle requires being an efficient information gatherer.', 
            'Repeat the cycle: guess, observe pattern, and calculate entropy.', 
            'Mastering information theory turns guessing into a precise science.'
        ]
        self.setup_layout("Conclusion: Information as a Tool", lecture_lines)

        # Colors for highlights
        HIGHLIGHT_COLOR_1 = "#FFFF00" # Yellow
        HIGHLIGHT_COLOR_2 = "#00FFFF" # Cyan
        HIGHLIGHT_COLOR_3 = "#00FF00" # Green

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR_1)
        
        # Create strategy cycle labels as requested
        guess_txt = Text("Guess", font_size=24, color=HIGHLIGHT_COLOR_1)
        observe_txt = Text("Observe", font_size=24, color=HIGHLIGHT_COLOR_1)
        calculate_txt = Text("Calculate", font_size=24, color=HIGHLIGHT_COLOR_1)
        repeat_txt = Text("Repeat", font_size=24, color=HIGHLIGHT_COLOR_1)

        # Applying position fixes (Issues 50, 51)
        self.place_in_area(guess_txt, "B3", "B4", scale_factor=0.8)
        self.place_in_area(observe_txt, "C5", "D5", scale_factor=0.8) # Issue 51
        self.place_in_area(calculate_txt, "E3", "E4", scale_factor=0.8)
        self.place_in_area(repeat_txt, "C2", "D2", scale_factor=0.8)  # Issue 50

        # Create White Arrows for circular flow
        arrow1 = CurvedArrow(guess_txt.get_right(), observe_txt.get_top(), color=WHITE, radius=2)
        arrow2 = CurvedArrow(observe_txt.get_bottom(), calculate_txt.get_right(), color=WHITE, radius=2)
        arrow3 = CurvedArrow(calculate_txt.get_left(), repeat_txt.get_bottom(), color=WHITE, radius=2)
        arrow4 = CurvedArrow(repeat_txt.get_top(), guess_txt.get_left(), color=WHITE, radius=2)

        cycle_group = VGroup(guess_txt, observe_txt, calculate_txt, repeat_txt, arrow1, arrow2, arrow3, arrow4)
        
        self.play(FadeIn(cycle_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR_2)
        
        self.play(FadeOut(cycle_group))

        # Asset Integration (Issue 35)
        # Load and place the Wordle SVG icon
        wordle_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/wordle.svg")
        self.place_at_grid(wordle_icon, "A4", scale_factor=0.5)

        # Wordle row colors
        W_GRAY = "#3A3A3C"
        W_YELLOW = "#B59F3B"
        W_GREEN = "#00FF00" 

        def create_wordle_row(colors):
            row = VGroup(*[Square(side_length=0.5, fill_opacity=1, fill_color=c, stroke_color=WHITE, stroke_width=1) for c in colors])
            row.arrange(RIGHT, buff=0.1)
            return row

        row1 = create_wordle_row([W_GRAY, W_YELLOW, W_GRAY, W_GRAY, W_GRAY])
        row2 = create_wordle_row([W_GREEN, W_GRAY, W_YELLOW, W_GRAY, W_GRAY])
        row3 = create_wordle_row([W_GREEN, W_GREEN, W_GREEN, W_GREEN, W_GREEN])

        grid_vgroup = VGroup(row1, row2, row3).arrange(DOWN, buff=0.1)
        # Expanded area to prevent squeezing (Issue 52)
        self.place_in_area(grid_vgroup, "B2", "E6")

        self.play(FadeIn(wordle_icon))
        # Animate rows quickly filling
        for row in grid_vgroup:
            self.play(FadeIn(row, shift=UP * 0.2), run_time=0.4)
            self.wait(0.1)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR_3)

        self.play(FadeOut(grid_vgroup), FadeOut(wordle_icon))

        info_text = Text("Information = Efficiency", font_size=32, color=HIGHLIGHT_COLOR_3)
        self.place_in_area(info_text, "C2", "D5")

        self.play(Write(info_text))
        self.wait(3)
