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
        self.setup_layout("Grouping the Beats: The Concept of the Bar", [
            "Music requires organization to stay together.",
            "We group beats into rhythmic measures.",
            "The downbeat signals a brand new cycle."
        ])
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Display three sets of four dots at #FFFFFF [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg].
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        dots_group = VGroup()
        for i in range(3):
            group = VGroup(*[Dot(color=WHITE) for _ in range(4)]).arrange(RIGHT, buff=0.3)
            dots_group.add(group)
        dots_group.arrange(DOWN, buff=0.5)
        
        # Fixing issue 39: place rhythmic_measures
        self.place_in_area(dots_group, 'A2', 'C5', scale_factor=0.6)
        
        self.play(FadeIn(dots_group), FadeIn(metronome.scale(0.5).to_edge(RIGHT)))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Group each set of four dots inside a rectangle at #FF00FF.
        rects = VGroup()
        for group in dots_group:
            rect = SurroundingRectangle(group, color="#FF00FF", buff=0.1)
            rects.add(rect)
        self.play(Create(rects))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the first dot of each group at #00FFFF [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/baton.svg].
        baton = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/baton.svg")
        # Fixing issue 40: downbeat_indicator
        self.place_at_grid(baton, 'D3', scale_factor=0.5)
        
        animations = [baton.animate.fade_to(WHITE, 0)]
        for group in dots_group:
            first_dot = group[0]
            animations.append(first_dot.animate.set_color("#00FFFF"))
        self.play(*animations)
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
