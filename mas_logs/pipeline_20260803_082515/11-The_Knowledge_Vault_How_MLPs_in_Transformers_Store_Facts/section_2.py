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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Information flows through layers via the residual stream.",
            "MLP layers read from this stream to process data.",
            "They write updates by adding new vectors back in."
        ]
        self.setup_layout("Prerequisite: The Residual Stream Conveyor Belt", lecture_lines)
        
        # Colors for lecture lines
        line1_color = BLUE_A
        line2_color = GREEN_A
        line3_color = YELLOW_A

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(line1_color)
        
        # Residual Stream as Conveyor Belt (Asset Integration - Issue #26)
        conveyor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/conveyor.svg", color=WHITE)
        self.place_in_area(conveyor, 'F1', 'F6', scale_factor=0.8)
        
        # Stream Label (Fixed position - Issue #31)
        stream_label = Text("Residual Stream", font_size=18, color=WHITE)
        self.place_at_grid(stream_label, 'D2', scale_factor=1.0)
        
        self.play(DrawBorderThenFill(conveyor))
        self.play(Write(stream_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(line2_color)
        
        # Vector box (Starting at F2 - Issue #32)
        vector_box = Rectangle(width=1.2, height=0.7, color=WHITE)
        vector_text = Text("Vector", font_size=18, color=WHITE)
        vector_group = VGroup(vector_box, vector_text)
        self.place_at_grid(vector_group, 'F2', scale_factor=0.8)
        
        # MLP Block (At B4 - Issue #33)
        mlp_block = Rectangle(width=2.5, height=1.2, color=line2_color)
        mlp_text = Text("MLP Block", font_size=24, color=line2_color)
        mlp_group = VGroup(mlp_block, mlp_text)
        self.place_at_grid(mlp_group, 'B4', scale_factor=1.0)
        
        self.play(FadeIn(vector_group))
        self.play(vector_group.animate.move_to(self.grid['F4']))
        
        # Read from stream
        read_arrow = Arrow(
            start=self.grid['F4'] + UP * 0.4, 
            end=self.grid['B4'] + DOWN * 0.6, 
            color=line2_color, 
            buff=0.1
        )
        
        self.play(FadeIn(mlp_group))
        self.play(GrowArrow(read_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(line3_color)
        
        # Write arrow (back from MLP to stream)
        # Position slightly offset to avoid overlapping the read arrow
        write_arrow = Arrow(
            start=self.grid['B4'] + DOWN * 0.6 + RIGHT * 0.3, 
            end=self.grid['F4'] + UP * 0.4 + RIGHT * 0.3, 
            color=line3_color, 
            buff=0.1
        )
        
        plus_sign = Text("+", font_size=32, color=line3_color)
        plus_sign.move_to(self.grid['F4'] + UP * 0.7 + RIGHT * 0.5)
        
        self.play(GrowArrow(write_arrow), Write(plus_sign))
        self.play(FadeOut(read_arrow), FadeOut(write_arrow))
        
        # Continue along stream
        self.play(vector_group.animate.move_to(self.grid['F6']))
        
        self.wait(2)
