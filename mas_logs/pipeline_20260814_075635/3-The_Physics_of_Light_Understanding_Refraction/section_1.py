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
        lecture_lines = [
            "Why does this pencil look broken in water?",
            "Light travels in a straight line usually.",
            "Entering water, the light path actually bends."
        ]
        self.setup_layout("Introduction: The 'Broken' Pencil Mystery", lecture_lines)
        
        # Assets
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        water = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg")
        
        # Adjust pencil for "broken" look
        pencil.shift(DOWN * 0.2 + RIGHT * 0.1)
        pencil_group = VGroup(glass, water, pencil)
        
        # Positioning from issue 26
        self.place_at_grid(pencil_group, 'D5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(FadeIn(pencil_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Issue 27 fix: Start from A6
        light_ray = DashedLine(self.grid['A6'], self.grid['D5'], color=WHITE)
        self.play(Create(light_ray))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        
        # Interface and break point as per storyboard
        interface = Line(LEFT*0.5, RIGHT*0.5, color="#FFD700").move_to(self.grid['C5'])
        bend_point = Dot(color="#FF4500").move_to(self.grid['C5'])
        
        # Show bending
        self.play(Create(interface), FadeIn(bend_point))
        self.wait(1)
