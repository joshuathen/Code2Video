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

class Section6Scene(TeachingScene):
    def construct(self):
        # Fetch data from storyboard/outline
        title_text = "Summary & Application: The Peak of the Mountain"
        lecture_lines = [
            "A zero derivative means the tangent is horizontal.",
            "This happens at the very peak of a mountain.",
            "Derivatives help us find the highest and lowest points."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors as per instructions/storyboard
        GREEN_M = "#00FF00" # Mountain
        WHITE_T = "#FFFFFF" # Tangent
        YELLOW_L = "#FFFF00" # Label
        
        # Define visual elements
        # Mountain Profile: Green curve peaking at the center
        mountain_path = VMobject(color=GREEN_M)
        # Points defining a hill shape relative to its own center
        mountain_points = [
            [-2.0, -1.5, 0],
            [-1.0, 0.0, 0],
            [0.0, 1.0, 0], # Peak
            [1.0, 0.0, 0],
            [2.0, -1.5, 0]
        ]
        mountain_path.set_points_as_corners(mountain_points).make_smooth()
        
        # Group and position the mountain in area C3-F5 (Issue 44 & 45 Fix)
        # This provides a more balanced vertical layout and leaves room for labels.
        mountain_group = VGroup(mountain_path)
        self.place_in_area(mountain_group, "C3", "F5", scale_factor=0.8)
        
        # Peak detection for the hiker and tangent line
        peak_point = mountain_path.get_top()
        
        # Hiker (represented as a small circle/dot as no asset was provided)
        hiker = Dot(color=ORANGE, radius=0.08)
        hiker.move_to(peak_point + UP * 0.1) # Position slightly above the peak
        
        # Tangent Line: Horizontal white line at the peak
        tangent_line = Line(LEFT, RIGHT, color=WHITE_T).scale(1.2)
        tangent_line.move_to(peak_point)
        
        # Slope Label: "Slope = 0" positioned near the peak
        # Per L003 and Issue 43 Fix: Use place_in_area for multi-word strings.
        # Positioned at B5-B6 to be safely above the tangent line (Issue 45 Fix).
        slope_label = Text("Slope = 0", font_size=24, color=YELLOW_L)
        self.place_in_area(slope_label, "B5", "B6", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # Line: "A zero derivative means the tangent is horizontal."
        self.play(self.lecture[0].animate.set_color(GREEN_M))
        self.play(Create(mountain_path), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "This happens at the very peak of a mountain."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE_T)
        )
        self.play(FadeIn(hiker))
        self.play(Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "Derivatives help us find the highest and lowest points."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW_L)
        )
        self.play(Write(slope_label))
        # Per storyboard: "Animate the text 'Slope = 0' flashing"
        self.play(Indicate(slope_label, color=YELLOW_L))
        self.wait(2)
