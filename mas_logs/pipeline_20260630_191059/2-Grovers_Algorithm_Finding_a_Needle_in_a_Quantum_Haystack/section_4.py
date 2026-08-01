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
        self.setup_layout(
            "Step 2: The Diffusion Operator (Inversion about the Mean)",
            [
                "Next, we apply the Grover diffusion operator.",
                "It performs an inversion about the average amplitude.",
                "The target's negative amplitude is pushed far above average.",
                "Meanwhile, all incorrect amplitudes are significantly reduced.",
                "This amplification makes the target state much more likely."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Display the bars from the previous step (one down, others up).
        self.lecture[0].set_color(WHITE)
        
        # Define bars
        bar_w = 0.6
        bar1 = Rectangle(width=bar_w, height=1.0, fill_opacity=0.8, color="#888888", fill_color="#888888")
        bar2 = Rectangle(width=bar_w, height=1.0, fill_opacity=0.8, color="#888888", fill_color="#888888")
        bar3 = Rectangle(width=bar_w, height=1.0, fill_opacity=0.8, color="#888888", fill_color="#888888")
        bar4 = Rectangle(width=bar_w, height=1.0, fill_opacity=0.8, color="#FFFF00", fill_color="#FFFF00")
        
        # Use columns 1-4 to avoid crowding the right margin (Fix for Issue 35)
        self.place_in_area(bar1, 'C1', 'D1')
        self.place_in_area(bar2, 'C2', 'D2')
        self.place_in_area(bar3, 'C3', 'D3')
        self.place_in_area(bar4, 'D4', 'E4')
        
        bars = VGroup(bar1, bar2, bar3, bar4)
        self.play(Create(bars))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a horizontal dashed line in #7CFC00 representing the 'Mean' amplitude.
        self.lecture[1].set_color("#7CFC00")
        
        # Mean line starts at col 1 and ends at col 5 (Fix for Issue 35)
        mean_y_start = (self.grid["C1"] + self.grid["D1"]) / 2
        mean_y_end = (self.grid["C5"] + self.grid["D5"]) / 2
        mean_line = DashedLine(start=mean_y_start, end=mean_y_end, color="#7CFC00")
        
        # Move mean label to C5 to avoid being cut off (Fix for Issue 34)
        mean_label = Text("Mean", font_size=18, color="#7CFC00")
        self.place_at_grid(mean_label, "C5", scale_factor=0.8)
        
        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The target's negative amplitude is pushed far above average.
        self.lecture[2].set_color("#FFFF00")
        
        # Target's distance indicator (Now at column 4)
        arrow_start = (self.grid["C4"] + self.grid["D4"]) / 2
        arrow_end = (self.grid["D4"] + self.grid["E4"]) / 2
        dist_arrow = DoubleArrow(
            start=arrow_start,
            end=arrow_end,
            buff=0,
            color=WHITE,
            stroke_width=2
        )
        
        self.play(Create(dist_arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Flip all bars across the mean line; the target bar becomes very tall.
        self.lecture[3].set_color("#888888")
        
        # Create final state bars at positions aligned with initial ones (Fix for Issue 36)
        new_bar1 = Rectangle(width=bar_w, height=0.01, fill_opacity=0.8, color="#888888", fill_color="#888888")
        new_bar2 = Rectangle(width=bar_w, height=0.01, fill_opacity=0.8, color="#888888", fill_color="#888888")
        new_bar3 = Rectangle(width=bar_w, height=0.01, fill_opacity=0.8, color="#888888", fill_color="#888888")
        new_bar4 = Rectangle(width=bar_w, height=2.0, fill_opacity=0.8, color="#FFFF00", fill_color="#FFFF00")
        
        self.place_at_grid(new_bar1, 'D1')
        self.place_at_grid(new_bar2, 'D2')
        self.place_at_grid(new_bar3, 'D3')
        self.place_in_area(new_bar4, 'B4', 'D4')
        
        self.play(
            Transform(bar1, new_bar1),
            Transform(bar2, new_bar2),
            Transform(bar3, new_bar3),
            Transform(bar4, new_bar4),
            FadeOut(dist_arrow),
            FadeOut(mean_line),
            FadeOut(mean_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the tall target bar in #00FF00.
        self.lecture[4].set_color("#00FF00")
        self.play(bar4.animate.set_color("#00FF00").set_fill("#00FF00"))
        self.play(Indicate(bar4, color="#00FF00"))
        self.wait(2)
