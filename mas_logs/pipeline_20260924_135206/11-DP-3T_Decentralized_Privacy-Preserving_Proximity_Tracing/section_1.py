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
            "We must track exposure without compromising user privacy.",
            "Public health safety versus individual anonymity.",
            "Alice tests positive, Bob was nearby."
        ]
        self.setup_layout("Introduction: The Privacy Paradox", lecture_lines)

        # Asset loading with try-except as per B010
        try:
            alice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg")
            alice.set_color("#FF6F61")
        except:
            alice = Dot(color="#FF6F61")
            
        try:
            bob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg")
            bob.set_color("#6B5B95")
        except:
            bob = Dot(color="#6B5B95")
            
        try:
            server = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")
            server.set_color("#88B04B")
        except:
            server = Square(color="#88B04B")
            
        q1 = Text("?", color="#EFC050")
        q2 = Text("?", color="#EFC050")

        # Initial positioning based on Critic/Orchestrator feedback
        self.place_at_grid(alice, "C2", scale_factor=0.5)
        self.place_at_grid(bob, "C5", scale_factor=0.5)
        self.place_at_grid(server, "E4", scale_factor=0.5)
        self.place_at_grid(q1, "B2", scale_factor=0.5)
        self.place_at_grid(q2, "B5", scale_factor=0.5)

        # Hiding them initially
        for obj in [alice, bob, server, q1, q2]:
            obj.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF6F61"))
        self.play(FadeIn(alice), FadeIn(bob))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#88B04B"))
        self.play(FadeIn(server))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#D64161"))
        # Animation: Alice moving toward Bob and changing to red glow
        alice_glow = Dot(color="#D64161").move_to(alice.get_center()).scale(2)
        self.play(FadeIn(alice_glow))
        self.play(alice.animate.move_to(bob.get_center() + LEFT * 0.5), run_time=1.5)
        self.play(FadeIn(q1), FadeIn(q2))
        
        # Flash
        self.play(Flash(alice, color=WHITE), Flash(bob, color=WHITE))
        self.wait(2)
