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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visual Proof: The Moving Accumulation Function", 
                          ["An accumulation function grows as it moves.", 
                           "Its rate of growth is the function's height.", 
                           "Thus, the derivative is the function itself."])
        
        # Setup Function and Area
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        axes.set_height(3)
        self.place_in_area(axes, 'B2', 'D4', scale_factor=0.9)
        
        func = axes.plot(lambda x: 0.5 * x + 1, x_range=[0, 4], color=BLUE)
        self.add(func)
        
        # Accumulation Area
        x_tracker = ValueTracker(0)
        area = always_redraw(lambda: axes.get_area(func, x_range=[0, x_tracker.get_value()], color="#00FFFF", opacity=0.5))
        self.add(area)
        
        # Vertical line
        line = always_redraw(lambda: DashedLine(
            axes.c2p(x_tracker.get_value(), 0), 
            axes.c2p(x_tracker.get_value(), func.underlying_function(x_tracker.get_value())),
            color=WHITE
        ))
        self.add(line)
        
        # Asset Placeholder (using placeholder for icon/none.svg)
        # Note: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # Adding a simple shape as a placeholder for the asset if it doesn't render
        icon = Dot(color=WHITE)
        self.place_at_grid(icon, 'A6')
        self.add(icon)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(x_tracker.animate.set_value(3.5), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        height_label = Text("h(x)", font_size=20, color="#FFFF00")
        self.place_at_grid(height_label, 'B4', scale_factor=0.8)
        self.play(FadeIn(height_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        result_text = Text("F'(x) = f(x)", font_size=24, color="#FF00FF")
        self.place_at_grid(result_text, 'F4', scale_factor=1.0)
        self.play(Write(result_text))
        self.wait(2)
