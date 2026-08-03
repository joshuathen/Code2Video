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

class Section2PrerequisiteKnowledgeScene(TeachingScene):
    def construct(self):
        # Data from storyboard and outline
        title_text = "Foundations: Chords and General Position"
        lecture_lines = [
            "A chord connects two points on the circle's edge.",
            "To maximize pieces, avoid three chords meeting together.",
            "Small shifts create extra regions in the middle."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Prepare the circle centered in area B2-E5
        circle = Circle(radius=2.0, color=WHITE)
        self.place_in_area(circle, "B2", "E5")
        circle_center = circle.get_center()

        def get_point(angle_deg):
            return circle_center + np.array([
                2.0 * np.cos(np.deg2rad(angle_deg)),
                2.0 * np.sin(np.deg2rad(angle_deg)),
                0
            ])

        # Bad chords (intersecting at center) - Color #FF0000
        p_bad = [get_point(a) for a in [0, 60, 120, 180, 240, 300]]
        chords_bad = VGroup(
            Line(p_bad[0], p_bad[3], color="#FF0000"),
            Line(p_bad[1], p_bad[4], color="#FF0000"),
            Line(p_bad[2], p_bad[5], color="#FF0000")
        )

        # Good chords (shifted slightly to create a central triangle) - Color #00FF00
        # We nudge the "opposite" points away from being true diameters
        p_good = [get_point(a) for a in [0, 60, 120, 170, 230, 290]]
        chords_good = VGroup(
            Line(p_good[0], p_good[3], color="#00FF00"),
            Line(p_good[1], p_good[4], color="#00FF00"),
            Line(p_good[2], p_good[5], color="#00FF00")
        )

        # Label for "Additional Region" - Color #FFFF00
        # Resolved Issue 28 & 29: Move label from A6 to C6-D6 area for better proximity and less crowding.
        label_text = Text("Additional Region", font_size=24, color="#FFFF00")
        self.place_in_area(label_text, "C6", "D6", scale_factor=0.6)
        
        # Arrow pointing to the central intersection area
        arrow = Arrow(label_text.get_left(), circle_center + np.array([0.1, 0.1, 0]), color="#FFFF00", buff=0.1)

        # === Animation for Lecture Line 1 ===
        # "A chord connects two points on the circle's edge."
        # Matching animation: Draw 3 chords intersecting at center (#FF0000)
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.play(Create(circle))
        self.play(Create(chords_bad))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "To maximize pieces, avoid three chords meeting together."
        # Matching animation: Slightly move chords to create a small central triangle (#00FF00)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )
        self.play(Transform(chords_bad, chords_good))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Small shifts create extra regions in the middle."
        # Matching animation: Label the central triangle as 'Additional Region' (#FFFF00)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        self.play(Write(label_text), Create(arrow))
        self.wait(2)

        # Final cleanup for the lecture highlighting
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
