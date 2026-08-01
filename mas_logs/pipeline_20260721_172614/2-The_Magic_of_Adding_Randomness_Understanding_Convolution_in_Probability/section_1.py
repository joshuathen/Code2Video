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

class Section1Scene(TeachingScene):
    def construct(self):
        # Define the lecture lines for the setup
        lecture_lines = [
            "Meet our Cyber-Cat, powered by two independent batteries.",
            "Each battery's life is a random variable, X and Y.",
            "If both provide 1 to 3 hours, what's the total?",
            "Is the probability just adding the averages together?",
            "Let's explore how these uncertainties combine through convolution."
        ]
        
        self.setup_layout("The Mystery of the Cyber-Cat's Battery", lecture_lines)
        
        # Define Colors
        COLOR_CAT = WHITE
        COLOR_BATTERY = GREEN
        COLOR_DIST = "#ADD8E6" # Light Blue
        COLOR_QUERY = YELLOW

        # === Animation for Lecture Line 1 ===
        # Meet our Cyber-Cat, powered by two independent batteries.
        self.lecture[0].set_color(COLOR_CAT)
        
        # Load cat asset
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        
        # Create a simple Battery icon
        bat_body = Rectangle(width=0.3, height=0.6, color=COLOR_BATTERY, fill_opacity=0.5)
        bat_tip = Rectangle(width=0.15, height=0.08, color=COLOR_BATTERY, fill_opacity=1.0).next_to(bat_body, UP, buff=0)
        battery_icon = VGroup(bat_body, bat_tip)
        
        self.place_at_grid(cat_icon, "B3", scale_factor=1.2)
        self.place_at_grid(battery_icon, "B4", scale_factor=1.2)
        
        self.play(FadeIn(cat_icon), FadeIn(battery_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Each battery's life is a random variable, X and Y.
        self.lecture[1].set_color(COLOR_DIST)
        
        # Transition cat/battery out
        self.play(FadeOut(cat_icon), FadeOut(battery_icon))
        
        # Battery X Distribution (Uniform 1-3)
        ax_a = Axes(
            x_range=[0, 4, 1], 
            y_range=[0, 1, 0.5], 
            axis_config={"include_tip": False, "font_size": 18},
            x_length=2.5,
            y_length=1.5
        ).add_coordinates()
        label_a = Text("Battery X", font_size=18, color=COLOR_DIST).next_to(ax_a, UP, buff=0.1)
        # Uniform distribution from 1 to 3 with height 0.5 (since area must be 1)
        dist_a = ax_a.plot_line_graph([0.5, 1, 1, 3, 3, 3.5], [0, 0, 0.5, 0.5, 0, 0], add_vertex_dots=False, line_color=COLOR_DIST)
        group_a = VGroup(ax_a, label_a, dist_a)
        
        # Battery Y Distribution (Uniform 1-3)
        ax_b = Axes(
            x_range=[0, 4, 1], 
            y_range=[0, 1, 0.5], 
            axis_config={"include_tip": False, "font_size": 18},
            x_length=2.5,
            y_length=1.5
        ).add_coordinates()
        label_b = Text("Battery Y", font_size=18, color=COLOR_DIST).next_to(ax_b, UP, buff=0.1)
        dist_b = ax_b.plot_line_graph([0.5, 1, 1, 3, 3, 3.5], [0, 0, 0.5, 0.5, 0, 0], add_vertex_dots=False, line_color=COLOR_DIST)
        group_b = VGroup(ax_b, label_b, dist_b)
        
        # Fix: Using updated grid positions from VideoCritic
        self.place_in_area(group_a, "C2", "D3", scale_factor=0.8)
        self.place_in_area(group_b, "C5", "D6", scale_factor=0.8)
        
        self.play(Create(ax_a), Create(ax_b))
        self.play(Write(label_a), Write(label_b), Create(dist_a), Create(dist_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # If both provide 1 to 3 hours, what's the total?
        self.lecture[2].set_color(COLOR_DIST)
        
        self.play(Indicate(dist_a), Indicate(dist_b))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Is the probability just adding the averages together?
        self.lecture[3].set_color(COLOR_QUERY)
        
        question_mark = Text("?", font_size=72, color=COLOR_QUERY)
        # Fix: Centered position based on VideoCritic feedback
        self.place_in_area(question_mark, "B2", "B5", scale_factor=1.2)
        
        self.play(Write(question_mark))
        self.play(Flash(question_mark, color=COLOR_QUERY, line_length=0.3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Let's explore how these uncertainties combine through convolution.
        self.lecture[4].set_color(WHITE)
        
        self.play(FadeOut(question_mark))
        
        conv_text = Text("Convolution", font_size=36, color=WHITE)
        # Fix: Centered position based on VideoCritic feedback
        self.place_in_area(conv_text, "B2", "B5", scale_factor=1.2)
        
        self.play(Write(conv_text))
        self.wait(2)
