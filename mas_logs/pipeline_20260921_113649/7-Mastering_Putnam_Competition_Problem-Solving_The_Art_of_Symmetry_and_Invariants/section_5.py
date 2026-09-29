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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Closing: Verification & Final Polish", [
            "Verify boundary conditions to ensure solution rigor.",
            "Check trivial cases for consistency and polish."
        ])
        
        # Checklist
        checklist = VGroup(
            Text("Boundary Check", font_size=24, color=WHITE),
            Text("Trivial Case Check", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(checklist, 'B4', scale_factor=0.6)
        
        # Asset Placeholder - If the file does not exist, use a placeholder shape
        try:
            key = ImageMobject("assets/golden_key_verification.png")
        except OSError:
            key = Star(color="#FFD700").scale(0.5)
        self.place_at_grid(key, 'C5', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(checklist[0]))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.play(FadeIn(checklist[1]))
        
        # Mark as done
        def create_checkmark(color=GREEN):
            return VGroup(Line(ORIGIN, 0.2*DOWN + 0.2*RIGHT), Line(0.2*DOWN + 0.2*RIGHT, 0.2*UP + 0.5*RIGHT)).set_color(color).scale(0.5)

        mark1 = create_checkmark(color=GREEN).next_to(checklist[0], RIGHT)
        mark2 = create_checkmark(color=GREEN).next_to(checklist[1], RIGHT)
        
        self.play(Create(mark1))
        self.play(self.lecture[0].animate.set_color(GREEN))
        self.play(Create(mark2))
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Final Message
        final_msg = Text("Problem Solved!", font_size=36, color="#FFD700")
        self.place_in_area(final_msg, 'E2', 'F5', scale_factor=0.9)
        self.play(FadeIn(final_msg), FadeIn(key))
        self.wait(2)
