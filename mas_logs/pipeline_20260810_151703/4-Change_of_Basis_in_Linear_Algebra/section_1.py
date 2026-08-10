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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Same vector, different observers.", "GPS vs local hiker coordinates.", "Think of a cat on a rug."]
        self.setup_layout("Intuitive Hook: The Viewpoint Shift", lecture_lines)
        
        # Load assets
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        gps_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gps.svg")
        
        # Define visual elements
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True})
        vector = Vector([1.5, 1.0], color=WHITE)
        cat = Dot(point=vector.get_end(), color=PINK)
        rug = Square(side_length=1.5, color=BLUE).rotate(PI/6)
        
        # Group them
        visuals = VGroup(axes, vector, cat, rug)
        # Applying instruction: self.place_in_area(visuals, 'B2', 'E5', scale_factor=0.6)
        self.place_in_area(visuals, 'B2', 'E5', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_fill(opacity=1), run_time=0.5)
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(camera_icon, 'A1', scale_factor=0.5)
        self.play(FadeIn(camera_icon), Create(axes), Create(vector), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_fill(opacity=1), run_time=0.5)
        self.lecture[1].set_color("#FFD700")
        self.place_at_grid(gps_icon, 'F6', scale_factor=0.5)
        self.play(vector.animate.set_color("#FFD700"), FadeIn(gps_icon), run_time=1.0)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_fill(opacity=1), run_time=0.5)
        self.lecture[2].set_color("#00FF00")
        self.play(Create(rug), cat.animate.set_color("#00FF00"), run_time=1.0)
        
        self.wait(2)
