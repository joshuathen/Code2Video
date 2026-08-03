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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout
        self.setup_layout(
            "The Iteration: Practice Makes Perfect",
            [
                "Learning requires repeating this cycle thousands of times.",
                "Each pass nudges the weights closer to perfection.",
                "Practice eventually transforms a random guess into accuracy."
            ]
        )

        # === Animation for Lecture Line 1 ===
        color_cycle = "#00BFFF"
        color_highlight = "#FFD700"
        
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(color_cycle))

        # Create cycle components
        forward_text = Text("Forward", font_size=20, color=WHITE)
        loss_text = Text("Loss", font_size=20, color=WHITE)
        backprop_text = Text("Backprop", font_size=20, color=WHITE)

        self.place_at_grid(forward_text, "A2")
        self.place_at_grid(loss_text, "A5")
        # Issue 36 fix: scale_factor=0.7
        self.place_in_area(backprop_text, "C3", "C4", scale_factor=0.7)

        cycle_group = VGroup(forward_text, loss_text, backprop_text)
        
        # Simple center for rotation
        center_pos = (self.grid["B3"] + self.grid["B4"]) / 2
        
        # Circular Arrow Representation
        circle_path = Circle(radius=1.5, color=color_cycle).move_to(center_pos)
        
        self.play(FadeIn(cycle_group), Create(circle_path))
        self.play(Rotate(circle_path, angle=TAU, about_point=center_pos), run_time=3, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_highlight)
        )

        # Create Graph for Error
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 1, 0.2],
            x_length=4,
            y_length=2.5,
            axis_config={"include_tip": True},
        )
        # Issue 35 fix: place in E1-F6, scale_factor=0.8
        self.place_in_area(axes, "E1", "F6", scale_factor=0.8)
        
        axes_label = Text("Error", font_size=18, color=WHITE).next_to(axes.y_axis, UP, buff=0.1)
        
        # Decreasing error curve
        error_curve = axes.plot(
            lambda x: 0.8 * np.exp(-0.5 * x),
            color=color_highlight,
            x_range=[0, 10]
        )

        self.play(Create(axes), Write(axes_label))
        self.play(Create(error_curve), run_time=4)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_highlight)
        )

        # "Golden Recipe" appears
        golden_recipe = Text("Golden Recipe", font_size=36, color=color_highlight)
        # Issue 34 fix: place in D2-D5, scale_factor=0.9
        self.place_in_area(golden_recipe, "D2", "D5", scale_factor=0.9)

        # Issue 22: Integrate recipe icon asset
        recipe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/recipe.svg")
        recipe_icon.set_color(color_highlight)
        # Place icon above the golden recipe text
        self.place_in_area(recipe_icon, "B3", "C4", scale_factor=0.8)

        self.play(
            FadeOut(cycle_group), 
            FadeOut(circle_path),
            FadeIn(golden_recipe),
            FadeIn(recipe_icon)
        )
        
        self.play(Indicate(golden_recipe, color=color_highlight), Indicate(recipe_icon, color=color_highlight))
        self.wait(2)
