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
        self.setup_layout("Summary and Real-World Impact", [
            "Flip, slide, multiply, and sum define the process.",
            "This operation powers modern AI and image recognition.",
            "Convolution is the bridge between data and meaning."
        ])
        
        # === Animation for Lecture Line 1 ===
        # 1D signal bars
        bars = VGroup(*[Rectangle(width=0.2, height=h, fill_opacity=0.8, color=BLUE) 
                       for h in [0.5, 1.2, 0.8, 1.5, 1.0]]).arrange(RIGHT, buff=0.1)
        self.place_in_area(bars, "A1", "B3", scale_factor=0.6)

        # 2D image grid
        grid_size = 4
        grid_2d = VGroup(*[Square(side_length=0.4, stroke_width=1, color=WHITE) 
                           for _ in range(grid_size**2)]).arrange_in_grid(rows=grid_size, buff=0)
        self.place_in_area(grid_2d, "A4", "B6", scale_factor=0.8)

        # Kernel sliding overlay
        kernel = Square(side_length=0.4, color=YELLOW, fill_opacity=0.4, stroke_width=2)
        kernel.move_to(grid_2d[0])

        self.play(
            self.lecture[0].animate.set_color(BLUE),
            Create(bars), 
            Create(grid_2d)
        )
        self.play(Create(kernel))
        # Slide kernel through the first row
        self.play(Succession(*[kernel.animate.move_to(grid_2d[i]) for i in range(1, 4)]), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color line 2
        self.play(self.lecture[1].animate.set_color(GOLD), run_time=1)
        
        cnn_text = Text("CNN: Convolutional Neural Networks", font_size=24, color="#FFD700")
        # Fix for Issue 32 & 34: C1-C6, scale 0.8
        self.place_in_area(cnn_text, "C1", "C6", scale_factor=0.8)
        
        self.play(Write(cnn_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3
        self.play(self.lecture[2].animate.set_color(GREEN), run_time=1)
        
        # Self-Driving Car icon (Asset integration)
        # Fix for Issue 20: Use Asset
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        # Fix for Issue 33 & 34: D1-F6, scale 1.2
        self.place_in_area(car, "D1", "F6", scale_factor=1.2)
        
        # Radar scan effect using a Sector
        radar_center = car.get_top()
        radar_scan = Sector(radius=1.5, angle=PI/4, start_angle=3*PI/8, 
                            color=GREEN, fill_opacity=0.2).move_to(radar_center, aligned_edge=DOWN)
        
        # Define the rotation animation using an updater
        angle_tracker = ValueTracker(0)
        def update_radar(m):
            # Oscillation between 3*PI/8 and 5*PI/8
            offset = np.sin(angle_tracker.get_value()) * (PI/8)
            m.set_start_angle(3*PI/8 + offset)
            # Re-align to car top in case of movement (though car is static)
            m.move_to(car.get_top(), aligned_edge=DOWN)

        radar_scan.add_updater(update_radar)
        
        self.play(FadeIn(car))
        self.add(radar_scan)
        self.play(angle_tracker.animate.set_value(4*PI), run_time=4, rate_func=linear)
        radar_scan.remove_updater(update_radar)
        
        self.wait(2)
