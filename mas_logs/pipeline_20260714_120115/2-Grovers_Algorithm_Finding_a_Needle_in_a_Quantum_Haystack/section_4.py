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
        # Setup title and lecture lines
        title_text = "Step 2: The Diffusion Operator (Inversion about the Mean)"
        lecture_lines = [
            "Next, we reflect all amplitudes around their average value.",
            "This \"inversion about the mean\" is a mathematical trick.",
            "The average is slightly lowered by the negative target amplitude.",
            "Reflecting makes the target amplitude much larger than the others.",
            "Other amplitudes shrink, boosting our chance of finding the target."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        BLUE_B = "#58C4DD"
        RED_B = "#F44336"
        MEAN_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # Grid positions for bars
        bar_cols = ["2", "3", "4", "5"]
        baseline_y = self.grid["D1"][1]
        
        # === Animation for Lecture Line 1 ===
        # Show the bars (one negative, three positive) and a horizontal dashed white (#FFFFFF) line for 'Mean'.
        self.lecture[0].set_color(YELLOW)
        
        bars = VGroup()
        for i in range(4):
            # Initial heights: positive states and the flipped target state from the oracle
            h = 0.6 if i < 3 else -0.6
            color = BLUE_B if i < 3 else RED_B
            bar = Rectangle(width=0.5, height=abs(h), fill_opacity=0.8, fill_color=color, stroke_width=2)
            pos = self.grid[f"D{bar_cols[i]}"]
            if h > 0:
                bar.move_to(pos + UP * (bar.height / 2))
            else:
                bar.move_to(pos + DOWN * (bar.height / 2))
            bars.add(bar)
            
        # Initial Mean line at 0.6 (level of the 3 positive bars)
        mean_line_y = baseline_y + 0.6
        mean_line = DashedLine(
            start=[self.grid["D2"][0] - 0.4, mean_line_y, 0],
            end=[self.grid["D5"][0] + 0.4, mean_line_y, 0],
            color=MEAN_COLOR
        )
        mean_label = Text("Mean", font_size=18, color=MEAN_COLOR)
        # Position label near the line in Row C. Applying fix for Issue 34: scale factor 0.6.
        self.place_at_grid(mean_label, 'C1', scale_factor=0.6)
        
        self.play(Create(bars), Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Shift the 'Mean' line down slightly to reflect the average of all current amplitudes.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Calculated new mean: (0.6 + 0.6 + 0.6 - 0.6) / 4 = 1.2 / 4 = 0.3
        new_mean_y = baseline_y + 0.3
        self.play(
            mean_line.animate.move_to([mean_line.get_center()[0], new_mean_y, 0]),
            mean_label.animate.move_to([mean_label.get_center()[0], new_mean_y, 0])
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reflect all bars across the dashed 'Mean' line: the target bar shoots up, others shrink.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Reflection formula: x_new = 2*mean - x_old
        # Positive bars: 2*0.3 - 0.6 = 0.0
        # Negative bar: 2*0.3 - (-0.6) = 1.2
        target_heights = [0.01, 0.01, 0.01, 1.2]
        
        transforms = []
        for i in range(4):
            new_h = target_heights[i]
            # Maintain bar width and color
            target_rect = Rectangle(width=0.5, height=new_h, fill_opacity=0.8, fill_color=bars[i].fill_color, stroke_width=2)
            # All bars are now positive or near-zero, so we move them UP from baseline
            target_rect.move_to(self.grid[f"D{bar_cols[i]}"] + UP * (target_rect.height / 2))
            transforms.append(Transform(bars[i], target_rect))
            
        self.play(*transforms)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight the now-tall target bar with a bright yellow (#FFFF00) glow.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Create a glow effect around the target bar
        glow = bars[3].copy().set_fill(HIGHLIGHT_COLOR, opacity=0.3).scale(1.2)
        self.play(FadeIn(glow))
        self.play(Indicate(bars[3], color=HIGHLIGHT_COLOR, scale_factor=1.05))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Display the text 'Inversion About the Mean' (#FFFFFF) at the top of the screen.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        inversion_text = Text("Inversion About the Mean", font_size=28, color=WHITE)
        # Position centered in the top of the chart area. Applying fix for Issue 33: A2-A5, scale 0.7.
        self.place_in_area(inversion_text, 'A2', 'A5', scale_factor=0.7)
        
        self.play(Write(inversion_text), FadeOut(glow))
        self.wait(2)
