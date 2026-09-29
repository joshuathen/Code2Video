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
        self.setup_layout("Verification and Conclusion", ["Data stays on user phones.", "Anonymity is maintained throughout.", "No identity is ever revealed."])
        
        # Elements
        phone = SVGMobject(path="/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg") if hasattr(self, "phone") else Rectangle(height=1.5, width=0.8, color=BLUE)
        lock = SVGMobject(path="/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg") if hasattr(self, "lock") else Dot(color=GREEN).scale(2)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(phone, "C2", scale_factor=0.8)
        self.play(DrawBorderThenFill(phone))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(lock, "C4", scale_factor=0.8)
        self.play(GrowFromCenter(lock))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color(YELLOW)
        
        # Cross animation centered on the lock, but path moved slightly
        cross = Line(start=UP*0.3+LEFT*0.3, end=DOWN*0.3+RIGHT*0.3, color=RED).move_to(lock.get_center())
        cross2 = Line(start=UP*0.3+RIGHT*0.3, end=DOWN*0.3+LEFT*0.3, color=RED).move_to(lock.get_center())
        
        # Using a group to animate the cross properly
        cross_group = VGroup(cross, cross2)
        self.play(Create(cross_group))
        self.wait(2)
