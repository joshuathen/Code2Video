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
        lecture_lines = ["Convolutions connect to system memory.", "They process historical inputs.", "Inputs affect the current system state."]
        self.setup_layout("Prerequisite Alignment: Why we use Convolutions", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Animate transition from linear transformation to convolution.
        line1_color = "#FFD700"  # Gold
        self.lecture[0].set_color(line1_color)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        # Fixing circle per issue 27/38
        circle = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.5)
        self.place_at_grid(circle, "B3", scale_factor=0.6)
        
        input_label = Text("Input Node", font_size=24, color=WHITE)
        self.place_at_grid(input_label, "B4", scale_factor=0.5)
        
        self.play(FadeIn(circle), Write(input_label))
        
        # === Animation for Lecture Line 2 ===
        # Show shift-invariance property visually with a moving pattern.
        line2_color = "#00FF7F"  # SpringGreen
        self.lecture[1].set_color(line2_color)
        
        # Fixing row_pattern position per issue 28/38
        row_pattern = VGroup(*[Square(side_length=0.4, color=WHITE).set_fill(GRAY, opacity=0.3) for _ in range(4)]).arrange(RIGHT, buff=0.1)
        self.place_in_area(row_pattern, "B1", "B4", scale_factor=0.6)
        
        # Fixing pointer position per issue 29/38
        pointer = Triangle(color=YELLOW).scale(0.2).rotate(PI)
        self.place_at_grid(pointer, "C1", scale_factor=0.7)
        
        self.play(FadeIn(row_pattern), FadeIn(pointer))
        self.play(pointer.animate.shift(RIGHT * 0.5), run_time=1.5)
        
        # === Animation for Lecture Line 3 ===
        # Display how convolution reuses weights across the input.
        line3_color = "#87CEFA"  # LightSkyBlue
        self.lecture[2].set_color(line3_color)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/processor.svg
        processor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/processor.svg")
        self.place_at_grid(processor_icon, "E3", scale_factor=0.8)
        
        label = Text("Weights Reused", font_size=20, color=WHITE)
        self.place_at_grid(label, "F3", scale_factor=0.8)
        self.play(FadeIn(processor_icon), Write(label))
        self.wait(1)
