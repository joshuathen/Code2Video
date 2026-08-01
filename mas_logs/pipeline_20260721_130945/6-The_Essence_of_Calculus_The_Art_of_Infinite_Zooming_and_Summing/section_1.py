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
        # Title and Lecture Lines for Section 1
        title = "The Big Picture: Static vs. Dynamic"
        lecture_lines = [
            "- Algebra handles objects at constant, steady speeds.",
            "- But our real world is constantly in motion.",
            "- Calculus is the mathematics of this continuous change."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Initially, ensure the first lecture line is the focus
        for i in range(1, len(self.lecture)):
            self.lecture[i].set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Line: "Algebra handles objects at constant, steady speeds."
        # Show a static photo frame of a Cheetah (#FFFFFF).
        
        photo_rect = Rectangle(width=3, height=2, color=WHITE)
        photo_text = Text("Cheetah: Static Frame", font_size=18, color=WHITE)
        photo_group = VGroup(photo_rect, photo_text).arrange(DOWN, buff=0.2)
        
        # Fix for Issue 29: Position up to B3-D5 and scale 0.8
        self.place_in_area(photo_group, 'B3', 'D5', scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            Create(photo_rect),
            Write(photo_text)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Line: "But our real world is constantly in motion."
        # The photo frame transforms into a video showing the Cheetah moving.
        
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Video representation
        video_rect = Rectangle(width=3, height=2, color=BLUE)
        video_text = Text("Cheetah: Dynamic Motion", font_size=18, color=BLUE)
        video_group = VGroup(video_rect, video_text).arrange(DOWN, buff=0.2)
        
        # Fix for Issue 30: Position up to B3-D5 and scale 0.8
        self.place_in_area(video_group, 'B3', 'D5', scale_factor=0.8)
        
        # Visual indicator of motion (moving lines inside the frame)
        motion_lines = VGroup(*[
            Line(LEFT, RIGHT, color=BLUE_A, stroke_width=2).scale(0.4) 
            for _ in range(3)
        ]).arrange(DOWN, buff=0.25).move_to(video_rect.get_center())
        
        # Use an updater to simulate movement (L008)
        def motion_updater(m):
            t = self.renderer.time
            for i, line in enumerate(m):
                # Animate horizontal position of speed lines
                line.set_x(video_rect.get_center()[0] + 0.4 * np.cos(t * 8 + i * 1.5))
        
        motion_lines.add_updater(motion_updater)
        
        self.play(
            Transform(photo_group, video_group),
            FadeIn(motion_lines)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line: "Calculus is the mathematics of this continuous change."
        # The words "Mathematics of Change" (#FFFF00) fade in above the Cheetah.
        
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        math_change_text = Text("Mathematics of Change", color="#FFFF00")
        
        # Fix for Issue 28: Position in Row A (A3-A6) for balance
        self.place_in_area(math_change_text, 'A3', 'A6', scale_factor=0.8)
        
        self.play(FadeIn(math_change_text))
        
        # Indicate the key concept (L004)
        self.play(Indicate(math_change_text))
        self.wait(3)
        
        # Cleanup
        motion_lines.remove_updater(motion_updater)
